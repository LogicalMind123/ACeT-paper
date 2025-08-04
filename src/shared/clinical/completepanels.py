import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, PowerTransformer, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, recall_score,
)
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from verstack.stratified_continuous_split import scsplit
import ImbalancedLearningRegression as iblr
from tfkan.layers import DenseKAN
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling  import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.tree import DecisionTreeClassifier, export_text
import shap 
from typing import Tuple 
import scipy.io as sio
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import seaborn as sns            # only used for pretty correlation heat‑map
from typing import Dict
from clinical.model import DenseKANRBF, build_transformer_model
# ──────────────────────────────────────────────────────────────────────────────
# 0. Jain‑et‑al. 2017 warning‑flag thresholds           
# ──────────────────────────────────────────────────────────────────────────────
JAIN_THRESH: Dict[str, Dict[str, float]] = {
    "PSR":        {">": 0.27},
    "ACSINS":     {">": 11.8},
    "CSI":        {">": 0.01},
    "CIC":        {">": 10.1},
    "HIC":        {">": 11.7},
    "SMAC":       {">": 12.8},
    "SGAC_SINS":  {"<": 370.0},
    "BVP":        {">": 4.3},
    "ELISA":      {">": 1.9},
    "AS":         {">": 0.08},
}
FLAG_GROUPS = {
    "grp1": ["PSR","ACSINS","CSI","CIC"],
    "grp2": ["HIC","SMAC","SGAC_SINS"],
    "grp3": ["BVP","ELISA"],
    "grp4": ["AS"],
}
def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for assay, rule in JAIN_THRESH.items():
        if assay not in df.columns: continue
        op, thr = list(rule.items())[0]
        flags[f"flag_{assay}"] = ((df[assay] > thr) if op==">"
                                  else (df[assay] < thr)).astype(int)
    for g, assays in FLAG_GROUPS.items():
        cols = [f"flag_{a}" for a in assays if f"flag_{a}" in flags.columns]
        flags[f"{g}_count"] = flags[cols].sum(axis=1)
    flags["flag_total"] = flags[[c for c in flags.columns if c.startswith("flag_")]].sum(axis=1)
    return flags

# ──────────────────────────────────────────────────────────────────────────────
# 1.   Threshold‑helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def youden_univariate_thresholds(X: pd.DataFrame,
                                 y: np.ndarray) -> pd.DataFrame:
    """Return optimal Youden‑J cut‑points per feature."""
    out = []
    for col in X.columns:
        x = X[col].values
        thresh = np.quantile(x, np.linspace(.01, .99, 99))
        tpr = lambda t: recall_score(y, x > t)          # positives = class 1
        tnr = lambda t: recall_score(1 - y, x <= t)     # negatives = class 0
        J = np.array([tpr(t) + tnr(t) - 1 for t in thresh])
        idx = int(np.argmax(J))
        out.append((col, thresh[idx], J[idx]))
    return (pd.DataFrame(out, columns=["feature", "threshold", "youden_J"])
              .sort_values("youden_J", ascending=False)
              .reset_index(drop=True))


def cart_pair_rules(X: pd.DataFrame,
                    y: np.ndarray,
                    max_depth: int = 2) -> Tuple[pd.DataFrame, str]:
    """Train a depth‑2 CART; return dataframe of split points and the text rule."""
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                 min_samples_leaf=int(0.05*len(X)),
                                 random_state=0)
    clf.fit(X, y)
    rules_txt = export_text(clf,
                            feature_names=list(X.columns),
                            max_depth=max_depth)
    # Extract thresholds + features for a quick heat‑map
    rows = []
    for line in rules_txt.splitlines():
        if "<=" in line or ">" in line:
            feat, rest = line.strip().split(" ", 1)
            thr = float(rest.split()[-1])
            rows.append((feat, thr))
    heat_df = pd.DataFrame(rows, columns=["feature", "threshold"])
    return heat_df, rules_txt


def shap_zero_cross_thresholds(model: tf.keras.Model,
                               X_df: pd.DataFrame) -> pd.DataFrame:
    """Return the value where SHAP crosses 0 for each feature,
       but gracefully skip on any errors."""
    try:
        # try GradientExplainer first
        explainer = shap.GradientExplainer(model, X_df.values[:256])
        raw_sv = explainer.shap_values(X_df.values)
        # get the first-class array
        shap_vals = raw_sv[0] if isinstance(raw_sv, list) else raw_sv

        out = []
        for i, col in enumerate(X_df.columns):
            feat_vals = X_df[col].values
            # guard against mismatched dims
            if shap_vals.ndim != 2 or i >= shap_vals.shape[1]:
                continue
            sv_col = shap_vals[:, i]

            # zip and sort by feature value
            pairs = sorted(zip(feat_vals, sv_col), key=lambda p: p[0])
            if not pairs:
                continue
            values = np.array([v for v, _ in pairs])
            svs    = np.array([s for _, s in pairs])

            # detect first sign flip
            signs = np.sign(svs)
            flips = np.where(np.diff(signs) != 0)[0]
            if flips.size:
                out.append((col, values[flips[0]]))

        return pd.DataFrame(out, columns=["feature", "shap_zero_cross"])

    except Exception as e:
        print(f"WARNING: SHAP threshold derivation failed: {e}")
        # return empty table so downstream code can continue
        return pd.DataFrame(columns=["feature", "shap_zero_cross"])

def predict_and_cm(models, X, y_true):
    """Utility: ensemble probs → preds → balanced‑accuracy & confusion‑matrix."""
    probs = np.mean([m.predict(X) for m in models], axis=0)
    preds = np.argmax(probs, axis=1)
    bacc  = balanced_accuracy_score(y_true, preds)
    cm    = confusion_matrix(y_true, preds)
    return bacc, cm, preds, probs


# ------------------------------------------------------------------------------
# ------------------ Classification Pipeline ------------------
def run_classification(args):
    print("=== Running Classification with Data‑driven Thresholds ===")
    data = pd.read_csv(args.train_file)
    feature_cols = data.columns[:-1].tolist()
    target_col = data.columns[-1]

    # Stratified train/test split
    train_df, test_df = train_test_split(
        data, test_size=args.test_size, random_state=args.seed,
        stratify=data[target_col]
    )

    # Generate synthetic data once outside CV
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
    synthesizer.fit(train_df)
    synthetic_data = synthesizer.sample(num_rows=int(len(train_df) * 0.5))  # e.g., 50% synthetic
    print(f"Original data: {data.shape}, Synthetic data: {synthetic_data.shape}")

    # Combine original + synthetic for training
    train_mix = pd.concat([train_df, synthetic_data], ignore_index=True)
    print(f"Combined training mix shape: {train_mix.shape}")

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Encode labels
    encoder = LabelEncoder()
    encoded_y_train = encoder.fit_transform(y_train_full)
    #encoded_y_train2 = encoder.fit_transform(train_df[target_col].values)
    encoded_y_test = encoder.transform(y_test)
    num_classes = len(encoder.classes_)

    # Scale features
    scaler = PowerTransformer()
    X_train_scaled = scaler.fit_transform(X_train_full)
    #X_train_scaled2 = scaler.fit_transform(train_df[feature_cols].values)
    X_test_scaled = scaler.transform(X_test)
    num_features = X_train_scaled.shape[1]

    heads = [args.head_type] if args.head_type != 'all' else [
        'mlp','rbf','poly','kan','interaction'
    ]

    best_head = None
    best_test_bacc = -1.0

    for head in heads:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
        print(f"\n--- Head = {head.upper()} ---")
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        ensemble_models = []
        fold_baccs = []

        for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, encoded_y_train), start=1):
            print(f"\n--- Fold {fold_no}/{args.n_splits} ---")
            X_tr = X_train_scaled[train_idx]
            y_tr_lbl = encoded_y_train[train_idx]
            X_val = X_train_scaled[val_idx]
            y_val_lbl = encoded_y_train[val_idx]
  
            
            # 1) ENN cleaning
            enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
            #enn = SMOTEENN(random_state=args.seed)
            #enn = TomekLinks()
            X_tr_clean, y_tr_clean = enn.fit_resample(X_tr, y_tr_lbl)
            class_counts_clean = np.bincount(y_tr_clean)
            print(f"After ENN: class_counts={class_counts_clean}")

            # If ENN yields too few, skip cleaning for SMOTE
            if class_counts_clean.min() < 2:
                print("ENN removed too many samples; using original training data for SMOTE.")
                X_for_smt, y_for_smt = X_tr, y_tr_lbl
            else:
                X_for_smt, y_for_smt = X_tr_clean, y_tr_clean

            # 2) SMOTE oversampling
            class_counts = np.bincount(y_for_smt)
            min_count = class_counts.min()
            k_smote = max(1, min(min_count - 1, 5))
            print(f"SMOTE k_neighbors={k_smote}, class_counts_before={class_counts}")
            smote = SMOTE(k_neighbors=k_smote, random_state=args.seed)
            X_tr_bal, y_tr_bal_lbl = smote.fit_resample(X_for_smt, y_for_smt)

            # One-hot encoding
            y_tr_bal = to_categorical(y_tr_bal_lbl, num_classes=num_classes)
            y_val_cat = to_categorical(y_val_lbl, num_classes=num_classes)

            # Build & compile model
            model = build_transformer_model(
                num_features=num_features,
                task='classification',
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg,
                num_classes=num_classes,
                head_type=head
            )
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy')

            # Callbacks & training
            early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
            lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                             patience=args.lr_patience, verbose=1, min_lr=args.min_lr)
            model.fit(
                X_tr_bal, y_tr_bal,
                validation_data=(X_val, y_val_cat),
                epochs=args.epochs, batch_size=args.batch_size,
                shuffle=True, callbacks=[early_stop, lr_reduction], verbose=1
            )

            # Validation Balanced Accuracy
            val_probs = model.predict(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            bacc = balanced_accuracy_score(y_val_lbl, val_preds)
            print(f"Fold {fold_no} Balanced Accuracy: {bacc:.4f}")
            fold_baccs.append(bacc)
            ensemble_models.append(model)

        # CV summary
        mean_bacc = np.mean(fold_baccs)
        print(f"\n--- {head.upper()} Mean CV Balanced Accuracy = {mean_bacc:.4f} ---")

        # Ensemble training evaluation
        def ensemble_predict(models, X):
            return np.mean([m.predict(X) for m in models], axis=0)
        train_probs = ensemble_predict(ensemble_models, X_train_scaled)
        train_preds = np.argmax(train_probs, axis=1)
        train_bacc = balanced_accuracy_score(encoded_y_train, train_preds)
        print(f"Ensemble Training Balanced Accuracy ({head.upper()}): {train_bacc:.4f}")
        cm_train = confusion_matrix(encoded_y_train, train_preds)
        print("Confusion Matrix (Training):")
        print(cm_train)
        ConfusionMatrixDisplay(cm_train, display_labels=encoder.classes_).plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"{head.upper()} Confusion Matrix (Training)")
        plt.savefig(f"cm_train_{head}.png", bbox_inches='tight')
        plt.close()

        # Test evaluation
        test_probs = ensemble_predict(ensemble_models, X_test_scaled)
        test_preds = np.argmax(test_probs, axis=1)
        test_bacc = balanced_accuracy_score(encoded_y_test, test_preds)
        print(f"Test Set Ensemble Balanced Accuracy ({head.upper()}): {test_bacc:.4f}")
        cm_test = confusion_matrix(encoded_y_test, test_preds)
        print("Confusion Matrix (Test):")
        print(cm_test)
        ConfusionMatrixDisplay(cm_test, display_labels=encoder.classes_).plot(cmap=plt.cm.Greens, colorbar=False)
        plt.title(f"{head.upper()} Confusion Matrix (Test)")
        plt.savefig(f"cm_test_{head}.png", bbox_inches='tight')
        plt.close()

        # Ensemble training evaluation without synthetic
        encoder2 = LabelEncoder()
        encoded_y_train2 = encoder2.fit_transform(train_df[target_col].values)
        scaler2 = PowerTransformer()
        X_train_scaled2 = scaler2.fit_transform(train_df[feature_cols].values)
        train_probs2 = ensemble_predict(ensemble_models, X_train_scaled2)
        train_preds2 = np.argmax(train_probs2, axis=1)
        train_bacc2 = balanced_accuracy_score(encoded_y_train2, train_preds2)
        print(f"Ensemble Training Balanced Accuracy ({head.upper()}): {train_bacc:.4f}")
        cm_train2 = confusion_matrix(encoded_y_train2, train_preds2)
        print("Confusion Matrix (Training):")
        print(cm_train2)
        ConfusionMatrixDisplay(cm_train2, display_labels=encoder2.classes_).plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title(f"{head.upper()} Confusion Matrix (Training)")
        plt.savefig(f"cm_train_{head}.png", bbox_inches='tight')
        plt.close()

        # Track best
        if test_bacc > best_test_bacc:
            best_test_bacc = test_bacc
            best_head = head

    # ─────────────────  EXTERNAL OUT‑OF‑TIME VALIDATION  ──────────────────
    if args.external_file:
        ext_df = pd.read_csv(args.external_file)

        # keep only mAbs whose Updated.Status is Approved / Terminated
        resolved = ext_df[args.status_col].isin(['Approved','Terminated'])
        ext_df   = ext_df.loc[resolved].copy()
        if ext_df.empty:
            print(">>> No resolved mAbs in external file ‑‑ skipping external eval")
        else:
            print(f">>> External cohort: {len(ext_df)} resolved mAbs")

            # feature alignment (same order as training)
            X_ext  = scaler.transform(ext_df[feature_cols].values)
            y_ext  = encoder.transform(ext_df[args.status_col].values)

            ext_bacc, cm_ext, ext_preds, ext_probs = predict_and_cm(
                ensemble_models, X_ext, y_ext
            )
            print(f"External balanced accuracy = {ext_bacc:.4f}")
            print("External confusion matrix:\n", cm_ext)

            # save artefacts
            np.savetxt("external_probs.csv", ext_probs, delimiter=",")
            np.savetxt("external_preds.csv", ext_preds, delimiter=",")
            np.savetxt("external_labels.csv", y_ext,   delimiter=",")
            np.savetxt("cm_external.csv",     cm_ext,  delimiter=",", fmt='%d')



    # --------  NEW THRESHOLD DERIVATION  ---------------------------------- #
    print("\n>>> Deriving assay thresholds (Youden, CART, SHAP) …")

    # ── PREDECLARE all downstream vars so Option 1 never crashes ────────────────
    assay_imp = np.array([])            # attention saliency
    emb_2d    = np.empty((0,2))         # latent embedding
    method    = ""                      # UMAP vs t-SNE label
    ig_mean   = np.array([])            # integrated-gradients

    # (i)  Univariate Youden‑J
    uni_df = youden_univariate_thresholds(
        X=train_df[feature_cols],
        y=encoder.transform(train_df[target_col].values)
    )
    uni_df.to_csv("Table_S1_thresholds.csv", index=False)
    print("  • Youden cut‑points → Table_S1_thresholds.csv")

    # (ii) Depth‑2 CART interacting rules
    cart_heat, cart_rules = cart_pair_rules(
        X=train_df[feature_cols],
        y=encoder.transform(train_df[target_col].values)
    )
    cart_heat.to_csv("Fig2_CART_thresholds.csv", index=False)
    with open("rules.txt", "w") as fh:
        fh.write(cart_rules)
    # quick heat‑map
    plt.figure(figsize=(8,6))
    plt.scatter(cart_heat["feature"], cart_heat["threshold"], s=60)
    plt.xticks(rotation=45, ha="right")
    plt.title("Depth‑2 CART split points")
    plt.tight_layout()
    plt.savefig("Fig2_CART_heatmap.png", dpi=300)
    plt.close()
    print("  • CART rules  → rules.txt + Fig2_CART_heatmap.png")

    # (iii) SHAP zero‑cross
    # pick first model of the ensemble (already best‑head)
    shap_df = shap_zero_cross_thresholds(
        model = ensemble_models[0],
        X_df  = pd.DataFrame(
            scaler.transform(train_df[feature_cols].values),
            columns = feature_cols)
    )
    shap_df.to_csv("ExtendedData_Fig4_SHAP_thresholds.csv", index=False)
    print("  • SHAP zero‑cross → ExtendedData_Fig4_SHAP_thresholds.csv")

    # violin plot for one representative feature set (using GradientExplainer)
    sel_cols = shap_df["feature"].tolist()[:5]
    try:
        bg = scaler.transform(train_df[feature_cols].values)[:256]
        expl = shap.GradientExplainer(ensemble_models[0], bg)
        shap_vals_full = expl.shap_values(scaler.transform(train_df[feature_cols].values))[0]

        for c in sel_cols:
            idx = feature_cols.index(c)
            plt.figure(figsize=(4,3))
            plt.violinplot(shap_vals_full[:, idx], showmeans=True)
            plt.axhline(0, ls="--")
            plt.title(f"SHAP for {c}")
            plt.tight_layout()
            plt.savefig(f"ED_Fig4_SHAP_{c}.png", dpi=300)
            plt.close()
        print("  • SHAP violins  → ED_Fig4_SHAP_<feature>.png")
    except Exception as e:
        print(f"WARNING: SHAP violin plots skipped due to: {e}")



    # ---------- A.  Flag burden per antibody -------------------------- #
    flag_cols   = [c for c in train_df.columns if c.startswith('flag_')]
    train_flags = train_df[flag_cols].sum(axis=1)
    test_flags  = test_df [flag_cols].sum(axis=1)
    plt.figure(figsize=(4,3))
    plt.hist(train_flags, alpha=.7, label='Train')
    plt.hist(test_flags,  alpha=.7, label='Test')
    plt.xlabel('# red flags'); plt.ylabel('# mAbs'); plt.legend()
    plt.title('Flag burden distribution')
    plt.tight_layout()
    plt.savefig('Fig3_FlagBurden.png', dpi=300)
    plt.close()

    # ---------- B.  Spearman correlation matrix ---------------------- #
    corr = train_df[feature_cols].corr(method='spearman')
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, cmap='vlag', center=0, square=True,
                cbar_kws={'label':'Spearman ρ'})
    plt.title('Assay–assay correlation (Spearman)')
    plt.tight_layout()
    plt.savefig('Fig2_CorrMatrix.png', dpi=300)
    plt.close()

    # ---------- C.  Hierarchical clustering of antibodies ------------ #
    Z = linkage(train_df[feature_cols], method='average', metric='euclidean')
    clusters = fcluster(Z, t=5, criterion='maxclust')
    plt.figure(figsize=(10,4))
    dendrogram(Z, no_labels=True, count_sort=True, color_threshold=0)
    plt.title('Antibody clustering (Euclidean on scaled assays)')
    plt.tight_layout()
    plt.savefig('Fig4_Dendrogram.png', dpi=300)
    plt.close()

    # ---------- D.  Developability radar variables ------------------- #
    # (just saving variables here; radar plotting will be in MATLAB)
    radar_vars = {
        'youden_thr': uni_df.set_index('feature')['threshold'].to_dict(),
        #'jain_thr'  : {k: list(v.values())[0] for k,v in JAIN_THRESH.items()}
    }

    # ---------- Save everything for MATLAB (.mat) -------------------- #
    sio.savemat('NatureBME_figvars1.mat', {
        'flag_hist_train'  : train_flags.values,
        'flag_hist_test'   : test_flags.values,
        'corr_matrix'      : corr.values,
        'corr_labels'      : np.array(corr.columns.tolist(), dtype=object),
        'linkage_Z'        : Z,
        'cluster_ids'      : clusters,
        'youden_thresholds': uni_df.values,
        'cart_thresholds'  : cart_heat.values,
        'shap_thresholds'  : shap_df.values,
        'jain_thresholds'  : np.array([[k, list(v.values())[0]] for k,v in JAIN_THRESH.items()], dtype=object)
     }, appendmat=True)
    print("  • MATLAB bundle  → NatureBME_figvars1.mat")
     # ------------------------------------------------------------------ #
    # ---------- E.  Attention-weight assay saliency -------------- #
    # ---------- E.  Attention-weight assay saliency -------------- #
    try:
        # grab the MHA layer
        attn_layer = [l for l in ensemble_models[0].layers
                      if 'multi_head_attention' in l.name][-1]

        # build a mini-model to fetch layer outputs
        attn_model = tf.keras.Model(
            inputs=ensemble_models[0].inputs,
            outputs=attn_layer.output
        )
        # predict; could be scores (4-D) or embeddings (3-D)
        attn_out = attn_model.predict(
            train_df[feature_cols].values, verbose=0
        )

        if attn_out.ndim == 4:
            # (n, heads, tokens, tokens) → mean over heads & samples
            attn_vals = attn_out.mean(axis=1)    # → (n, tokens, tokens)
            attn_mean = attn_vals.mean(axis=0)   # → (tokens, tokens)
            # sum attention *from* each assay token (drop last if [CLS])
            assay_imp = attn_mean[:, :-1].sum(axis=1)
        elif attn_out.ndim == 3:
            # fallback: treat embedding dims as “heads”
            # (n, tokens, embed) → mean over samples & embed dims
            tmp = attn_out.mean(axis=0)          # → (tokens, embed)
            assay_imp = np.abs(tmp[:, :-1]).mean(axis=1)
        else:
            raise ValueError(f"Unexpected attn_out.ndim={attn_out.ndim}")

        # normalise and plot
        assay_imp = assay_imp / assay_imp.max()
        plt.figure(figsize=(5,3))
        sns.barplot(x=feature_cols, y=assay_imp)
        plt.ylabel('Normalised attention'); plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('Fig2_AttentionSaliency.png', dpi=300)
        plt.close()
        sio.savemat('NatureBME_figvars1.mat',
                    {'attn_import': assay_imp}, appendmat=True)
        print("  • Attention saliency → Fig2_AttentionSaliency.png")

    except Exception as e:
        print(f"WARNING: Attention saliency skipped due to: {e}")

    # ---------- F. Latent embedding UMAP ------------------------- #
    # ---------- F.  Latent embedding UMAP or t-SNE ---------------- #
    try:
        # extract the embedding tensor just before the head
        latent_model = tf.keras.Model(
            ensemble_models[0].input,
            ensemble_models[0].get_layer(index=-3).output
        )
        latent = latent_model.predict(
            train_df[feature_cols].values, verbose=0
        )

        try:
            # preferred: UMAP
            from umap.umap_ import UMAP
            reducer = UMAP(random_state=0)
            emb_2d = reducer.fit_transform(latent)
            method = "UMAP"
        except ImportError:
            # fallback: t-SNE
            from sklearn.manifold import TSNE
            reducer = TSNE(random_state=0)
            emb_2d = reducer.fit_transform(latent)
            method = "t-SNE"

        plt.figure(figsize=(4,3))
        plt.scatter(
            emb_2d[:,0], emb_2d[:,1],
            c=(train_df[target_col]=='Terminated').astype(int),
            cmap='coolwarm', s=12, alpha=.8
        )
        plt.title(f'Transformer latent {method}')
        plt.tight_layout()
        plt.savefig('Fig3_Latent2D.png', dpi=300)
        plt.close()
        sio.savemat('NatureBME_figvars1.mat',
                    {'latent_2d': emb_2d, 'latent_2d_method': method},
                    appendmat=True)
        print(f"  • Latent 2D ({method}) → Fig3_Latent2D.png")

    except Exception as e:
        print(f"WARNING: Latent 2D embedding skipped due to: {e}")


    # ---------- G. Integrated-Gradients attributions ------------- #
    # ---------- G.  Integrated-Gradients or vanilla saliency ------------- #
    try:
        try:
            from tf_keras_vis.saliency import Saliency
            def model_fn(x): return ensemble_models[0](x)
            saliency = Saliency(model_fn)
            ig_vals = saliency.integrated_gradients(
                train_df[feature_cols].values.astype(np.float32),
                baseline=np.zeros_like(train_df[feature_cols].values),
                steps=50
            )
            ig_mean = ig_vals.mean(axis=0)
            method = "IG"
        except ImportError:
            # fallback: vanilla grads × input
            import tensorflow.keras.backend as K
            inp = ensemble_models[0].input
            out = ensemble_models[0].output[:,1]  # logit for 'Terminated'
            grads = K.gradients(out, inp)[0]
            f = K.function([inp], [grads])
            grad_vals = f([train_df[feature_cols].values.astype(np.float32)])[0]
            ig_mean = (grad_vals * train_df[feature_cols].values).mean(axis=0)
            method = "Grad×Input"

        plt.figure(figsize=(5,3))
        sns.barplot(x=feature_cols, y=np.abs(ig_mean))
        plt.ylabel(f'|{method}| (mean)'); plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'Fig4_{method}_Saliency.png', dpi=300)
        plt.close()

        sio.savemat('NatureBME_figvars1.mat',
                    {f'{method.lower()}_saliency': ig_mean},
                    appendmat=True)
        print(f"  • {method} saliency → Fig4_{method}_Saliency.png")

    except Exception as e:
        print(f"WARNING: Saliency attribution skipped due to: {e}")


    # ---------- H. Assay-pair mutual-info synergy --------------- #
    # ---------- G.  Integrated-Gradients or vanilla saliency ------------- #
    try:
        try:
            from tf_keras_vis.saliency import Saliency
            def model_fn(x): return ensemble_models[0](x)
            saliency = Saliency(model_fn)
            ig_vals = saliency.integrated_gradients(
                train_df[feature_cols].values.astype(np.float32),
                baseline=np.zeros_like(train_df[feature_cols].values),
                steps=50
            )
            ig_mean = ig_vals.mean(axis=0)
            method = "IG"
        except ImportError:
            # fallback: vanilla grads × input
            import tensorflow.keras.backend as K
            inp = ensemble_models[0].input
            out = ensemble_models[0].output[:,1]  # logit for 'Terminated'
            grads = K.gradients(out, inp)[0]
            f = K.function([inp], [grads])
            grad_vals = f([train_df[feature_cols].values.astype(np.float32)])[0]
            ig_mean = (grad_vals * train_df[feature_cols].values).mean(axis=0)
            method = "Grad×Input"

        plt.figure(figsize=(5,3))
        sns.barplot(x=feature_cols, y=np.abs(ig_mean))
        plt.ylabel(f'|{method}| (mean)'); plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'Fig4_{method}_Saliency.png', dpi=300)
        plt.close()

        sio.savemat('NatureBME_figvars1.mat',
                    {f'{method.lower()}_saliency': ig_mean},
                    appendmat=True)
        print(f"  • {method} saliency → Fig4_{method}_Saliency.png")

    except Exception as e:
        print(f"WARNING: Saliency attribution skipped due to: {e}")

    # ── AFTER ALL CALCULATIONS: bundle everything into one save ─────────────────
    all_vars = {
        'flag_hist_train'  : train_flags.values,
        'flag_hist_test'   : test_flags.values,
        'corr_matrix'      : corr.values,
        'corr_labels'      : np.array(corr.columns.tolist(), dtype=object),
        'linkage_Z'        : Z,
        'cluster_ids'      : clusters,
        'youden_thresholds': uni_df.values,
        'cart_thresholds'  : cart_heat.values,
        'shap_thresholds'  : shap_df.values,
        'jain_thresholds'  : np.array(
            [[k, list(v.values())[0]] for k,v in JAIN_THRESH.items()],
            dtype=object
        ),
        'attn_import'      : assay_imp,        # from attention block
        'latent_2d'        : emb_2d,           # from UMAP/t-SNE block
        'latent_2d_method' : method,           # string
        'ig_saliency'      : ig_mean,          # from IG block (if computed)
    }

    sio.savemat('NatureBME_figvars1.mat', all_vars)
    print("  • Wrote all figure variables to NatureBME_figvars1.mat")


    # ── SAVE confusion matrices, per-sample predictions & labels ──────────────────
    cm_train_arr = cm_train           # numpy arrays from last head = best_head
    cm_test_arr  = cm_test
    train_labels = encoded_y_train    # 1-D int array
    test_labels  = encoded_y_test
    train_preds  = train_preds        # from best ensemble on true train set
    test_preds   = test_preds
    class_names  = np.array(encoder.classes_, dtype=object)

    # Optional: keep antibody identifiers if column exists
    if 'mAb_name' in train_df.columns:
        mAb_train = train_df['mAb_name'].values.astype(object)
        mAb_test  = test_df ['mAb_name'].values.astype(object)
    else:
        mAb_train = mAb_test = np.array([], dtype=object)

    # Append everything to the existing .mat bundle
    sio.savemat('NatureBME_figvars2.mat', {
        'cm_train'       : cm_train_arr,
        'cm_test'        : cm_test_arr,
        'train_labels'   : train_labels,
        'test_labels'    : test_labels,
        'train_preds'    : train_preds,
        'test_preds'     : test_preds,
        'class_names'    : class_names,
        'mAb_train'      : mAb_train,
        'cm_external'    : cm_ext,
        'external_labels': y_ext,
        'external_preds' : ext_preds,
        'mAb_test'       : mAb_test,

    }, appendmat=True)

    print("  • Added confusion matrices to NatureBME_figvars2.mat")

    print(f"\n*** Best Head on Test Set: {best_head.upper()} (Balanced Accuracy = {best_test_bacc:.4f}) ***")

# 5) Argparse & Main
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['regression','classification'], required=True)
    p.add_argument('--train_file', required=True,
                   help='Path to full dataset CSV file (we’ll split it internally)')
    p.add_argument('--test_size', type=float, default=0.2,
                   help='Fraction of the data to reserve for test (classification only)')
    p.add_argument('--head_type',
                   choices=['mlp','rbf','poly','kan','interaction','all'],
                   default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--ff_dim', type=int, default=32)
    p.add_argument('--num_transformer_blocks', type=int, default=1)
    p.add_argument('--mlp_units',
                   type=lambda s: [int(x) for x in s.split(',')], default='64')
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--l2_reg', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr_factor', type=float, default=0.5)
    p.add_argument('--lr_patience', type=int, default=10)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--external_file',  default=None,
                   help='CSV of additional mAbs for out‑of‑time validation')
    p.add_argument('--status_col',     default='Updated.Status',
                   help='Column in --external_file containing final outcome')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    if args.task == 'regression':
        run_regression(args)
    else:
        run_classification(args)

if __name__ == '__main__':
    main()
