
import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, PowerTransformer, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from verstack.stratified_continuous_split import scsplit
import ImbalancedLearningRegression as iblr
from tfkan.layers import DenseKAN
import shap
from scipy.io import savemat
from sklearn.base import BaseEstimator, RegressorMixin

from viscosity.model import DenseKANRBF, build_transformer_model
# ------------------------------------------------------------------------------
# 3) Regression Pipeline (uses train.csv and test.csv)
# ------------------------------------------------------------------------------
def run_regression(args):
    print("=== Running Regression Task ===")
    train = pd.read_csv(args.train_file)
    test  = pd.read_csv(args.test_file)
    target_col = train.columns[-1]

    # 1) generate synthetic data & combine
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = SingleTableMetadata(); metadata.detect_from_dataframe(train)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
    synthesizer.fit(train)
    synthetic_data = synthesizer.sample(num_rows=26)

    train_mix = pd.concat([train, train, synthetic_data], ignore_index=True)
    
    print("Original train shape:", train.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    print("Train_mix shape:", train_mix.shape)
    print(train_mix.tail())

    # 2) global feature‐scaling (we'll rebalance per fold below)
    sc_X = QuantileTransformer()
    X_all = sc_X.fit_transform(train_mix.iloc[:, :-1].values)
    X_test = sc_X.transform(test.iloc[:, :-1].values)
    y_all = train_mix.iloc[:, -1].values
    feature_cols = train_mix.columns[:-1]

    # 3) set up CV & model heads
    heads = [args.head_type] if args.head_type!='all' else ['mlp','rbf','spline','kan','interaction']
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for head in heads:
        print(f"--- Head = {head.upper()} ---")
        ens_models, val_losses = [], []
        cv_r2, cv_rmse, cv_mae, cv_spearman = [], [], [], []

        for fold, (tr, vl) in enumerate(kf.split(X_all, y_all), 1):
            print(f"Fold {fold} (head={head})")

            # --- 3a) Extract this fold's training data ---
            X_tr, y_tr = X_all[tr], y_all[tr]

            # --- 3b) Build a DataFrame so iblr can work on it ---
            df_tr = pd.DataFrame(X_tr, columns=feature_cols)
            df_tr[target_col] = y_tr

            # --- 3c) ENN cleaning + SMOTE oversampling on this fold only ---
            df_tr_clean = iblr.enn(data=df_tr, y=target_col, rel_coef=0.5)

            if df_tr_clean.isnull().values.any():
               df_tr_clean = df_tr_clean.dropna()
            
            df_tr_clean = df_tr_clean.loc[:, df_tr_clean.nunique() > 1]

            # 3) Try SMOTE with a fallback
            try:
               df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=0.5)
            except ValueError as e:
               print(f"[Fold {fold}] SMOTE failed ({e}); retrying with rel_coef=0.25")
            try:
               df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=0.25)
            except ValueError as e2:
               print(f"[Fold {fold}] SMOTE retry failed ({e2}); skipping SMOTE")
            
            df_tr_bal = df_tr_clean.copy()
            #df_tr_bal = df_tr.copy()

            X_tr_bal = df_tr_bal.iloc[:, :-1].values
            y_tr_bal = df_tr_bal[target_col].values

            # --- 3d) build & compile model ---
            m = build_transformer_model(
                num_features=X_all.shape[1],
                task='regression',
                head_type=head,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                ff_dim=args.ff_dim,
                num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units,
                dropout_rate=args.dropout_rate,
                l2_reg=args.l2_reg
            )
            m.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss='log_cosh'
            )

            # --- 3e) scale the fold's target and train ---
            sc_y = QuantileTransformer()
            y_tr_s = sc_y.fit_transform(y_tr_bal.reshape(-1,1)).flatten()
            y_vl_s = sc_y.transform(y_all[vl].reshape(-1,1)).flatten()

            cb = [
                EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
            ]

            m.fit(
                X_tr_bal, y_tr_s,
                validation_data=(X_all[vl], y_vl_s),
                epochs=args.epochs, batch_size=args.batch_size,
                callbacks=cb, verbose=1
            )

            # --- 3f) record validation loss & model ---
            vloss = m.evaluate(X_all[vl], y_vl_s, verbose=0)
            ens_models.append((m, sc_y))   # keep sc_y for each model if you want per‐fold inverse later
            val_losses.append(vloss)

            print(f"CV losses for {head}: {val_losses}")

            # compute validation metrics
            preds_vl_s = m.predict(X_all[vl]).flatten()
            preds_vl = sc_y.inverse_transform(preds_vl_s.reshape(-1,1)).flatten()
            true_vl = y_all[vl]
            cv_r2.append(r2_score(true_vl, preds_vl))
            cv_rmse.append(mean_squared_error(true_vl, preds_vl, squared=False))
            cv_mae.append(mean_absolute_error(true_vl, preds_vl))
            coef, _ = spearmanr(true_vl, preds_vl)
            cv_spearman.append(coef)

        # print aggregated CV metrics
        print(f"{head.upper()} CV metrics:")
        print(f"  R²       = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")
        print(f"  RMSE     = {np.mean(cv_rmse):.3f} ± {np.std(cv_rmse):.3f}")
        print(f"  MAE      = {np.mean(cv_mae):.3f} ± {np.std(cv_mae):.3f}")
        print(f"  Spearman = {np.mean(cv_spearman):.3f} ± {np.std(cv_spearman):.3f}\n")

        # --- 4) ensemble‐predict & inverse‐transform ---
        preds_s = np.mean([mdl.predict(X_test).flatten() for mdl, _ in ens_models], axis=0)
        # NOTE: here we’re using the LAST fold’s sc_y; 
        # for full correctness you might want to inverse‐transform each mdl’s preds with its own sc_y
        preds = ens_models[-1][1].inverse_transform(preds_s.reshape(-1,1)).flatten()

        # --- 5) final metrics ---
        trues = test.iloc[:, -1].values
        # ---- Analysis and Export Section ---- #

        # Evaluate final metrics
        r2   = r2_score(trues, preds)
        rmse = mean_squared_error(trues, preds, squared=False)
        mae  = mean_absolute_error(trues, preds)
        spearman_coef, _ = spearmanr(trues, preds)

        print("\n--- Final Test Set Metrics ---")
        print(f"R²       : {r2:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"Spearman : {spearman_coef:.4f}")

        # Save predictions to CSV for MATLAB
        pred_df = pd.DataFrame({
            'True_Viscosity': trues,
            'Predicted_Viscosity': preds
        })
        pred_df.to_csv('viscosity_predictions.csv', index=False)
        print("Saved predictions to 'viscosity_predictions.csv'.")

        # Scatter plot Predicted vs True (log-log)
        plt.figure(figsize=(8,8))
        plt.scatter(trues, preds, alpha=0.7, c='crimson')
        plt.xscale('log')
        plt.yscale('log')
        min_val = min(trues.min(), preds.min())
        max_val = max(trues.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        plt.xlabel('True Viscosity', fontsize=14)
        plt.ylabel('Predicted Viscosity', fontsize=14)
        plt.title('Predicted vs True Viscosity (log-log)', fontsize=16)
        plt.grid(True)
        plt.savefig('viscosity_pred_vs_true.png', bbox_inches='tight')
        plt.close()
        print("Saved scatter plot 'viscosity_pred_vs_true.png'.")

        # Permutation Feature Importance
        from sklearn.base import BaseEstimator, RegressorMixin

        class EnsembleRegressor(BaseEstimator, RegressorMixin):
            def __init__(self, models, scaler):
                self.models = models
                self.scaler = scaler

            def fit(self, X, y=None):
                # Dummy fit method (does nothing since models are pre-trained)
                return self

            def predict(self, X):
                preds_s = np.mean([m.predict(X).flatten() for m, _ in self.models], axis=0)
                return self.scaler.inverse_transform(preds_s.reshape(-1,1)).flatten()

        ensemble_regressor = EnsembleRegressor(ens_models, sc_y)
        perm_importance = permutation_importance(
            ensemble_regressor, X_test, trues, n_repeats=10, scoring='r2', random_state=args.seed
        )
        feature_importances = perm_importance.importances_mean
        sorted_idx = np.argsort(feature_importances)[::-1]

        # Feature Importance plot
        plt.figure(figsize=(10,6))
        plt.bar(np.array(feature_cols)[sorted_idx], feature_importances[sorted_idx], color='teal')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Importance (Mean decrease in R²)", fontsize=14)
        plt.title("Permutation Feature Importance", fontsize=16)
        plt.tight_layout()
        plt.savefig('viscosity_feature_importance.png')
        plt.close()
        print("Saved feature importance plot 'viscosity_feature_importance.png'.")

        # SHAP values (explainability)
        explainer = shap.KernelExplainer(ensemble_regressor.predict, X_test[:min(100, len(X_test))])
        shap_values = explainer.shap_values(X_test, nsamples=100)

        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.savefig('shap_summary_viscosity.png', bbox_inches='tight')
        plt.close()
        print("Saved SHAP summary plot 'shap_summary_viscosity.png'.")

        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
        plt.savefig('shap_bar_viscosity.png', bbox_inches='tight')
        plt.close()
        print("Saved SHAP bar plot 'shap_bar_viscosity.png'.")

        # Export MATLAB workspace variables (fully matching codeviso.txt structure)
        matlab_vars = {
            'dataMatrix': np.hstack((X_test, trues.reshape(-1,1), preds.reshape(-1,1))),
            'varNames': np.array(list(feature_cols) + ['True_Viscosity', 'Predicted_Viscosity'], dtype=object),
            'y_test': trues.reshape(-1, 1),
            'y_pred': preds.reshape(-1, 1),
            'featureImportance': feature_importances.reshape(-1, 1),
            'featureNames': np.array(feature_cols, dtype=object),
            'r2_transformer': np.array([[r2]]),
            'rmse_transformer': np.array([[rmse]]),
            'mae_transformer': np.array([[mae]]),
            'spearman_transformer': np.array([[spearman_coef]])
        }

        # Add these two lines:
        matlab_vars['shapValues']     = shap_values       # shape (n_test, n_features)
        matlab_vars['shapImportance'] = np.abs(shap_values).mean(axis=0).reshape(-1,1)

        # Finally save:
        savemat('viscosity_results.mat', matlab_vars)
        print("Exported MATLAB variables, including SHAP, to 'viscosity_results.mat'.")

# ------------------------------------------------------------------------------
# 5) Argparse & Main
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['regression','classification'], required=True)
    p.add_argument('--train_file', required=True, help='Path to training CSV file')
    p.add_argument('--test_file',  required=True, help='Path to test CSV file')
    p.add_argument('--head_type', choices=['mlp','rbf','spline','kan','interaction','all'], default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--ff_dim', type=int, default=32)
    p.add_argument('--num_transformer_blocks', type=int, default=1)
    p.add_argument('--mlp_units', type=lambda s: [int(x) for x in s.split(',')], default='64')
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--l2_reg', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr_factor', type=float, default=0.5)
    p.add_argument('--lr_patience', type=int, default=10)
    p.add_argument('--min_lr', type=float, default=1e-6)
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
