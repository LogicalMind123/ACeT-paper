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
from scipy.io import savemat

# ------------------------------------------------------------------------------
# 3) Regression Pipeline (uses train.csv and test.csv)
# ------------------------------------------------------------------------------

def run_regression(args):
    print("=== Running Regression Task ===")
    train = pd.read_csv(args.train_file)
    test  = pd.read_csv(args.test_file)
    feature_cols = list(train.columns[:-1])
    target_col   = train.columns[-1]

    # 1) generate synthetic data & combine
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata     = SingleTableMetadata(); metadata.detect_from_dataframe(train)
    synthesizer  = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
    synthesizer.fit(train)
    synthetic_data = synthesizer.sample(num_rows=84)
    train_mix    = pd.concat([train, train, synthetic_data], ignore_index=True)

    print("Original train shape:", train.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    print("Train_mix shape:", train_mix.shape)
    print(train_mix.tail())

    # 2) global feature‐scaling
    from sklearn.preprocessing import QuantileTransformer
    sc_X = QuantileTransformer()
    X_all = sc_X.fit_transform(train_mix.iloc[:, :-1].values)
    X_test = sc_X.transform(test.iloc[:, :-1].values)
    y_all = train_mix.iloc[:, -1].values
    y_test = test.iloc[:, -1].values

    # 3) set up CV & storage for metrics
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    head = 'rbf'
    print(f"--- Head = {head.upper()} ---")

    # Prepare dicts to store fold‐level CV metrics (added nrmse & nmae)
    cv_metrics = {
        'Transformer': {'r2': [], 'rmse': [], 'mae': [], 'nrmse': [], 'nmae': []},
        'Ridge':       {'r2': [], 'rmse': [], 'mae': [], 'nrmse': [], 'nmae': []},
        'SVR':         {'r2': [], 'rmse': [], 'mae': [], 'nrmse': [], 'nmae': []},
        'RandomForest':{'r2': [], 'rmse': [], 'mae': [], 'nrmse': [], 'nmae': []}
    }
    # Storage for test‐set results (will also hold nrmse & nmae)
    test_metrics = {
        'Transformer': {},
        'Ridge': {},
        'SVR': {},
        'RandomForest': {}
    }

    # a) Transformer (KAN head) CV and test
    transformer_models = []
    for fold, (tr, vl) in enumerate(kf.split(X_all, y_all), 1):
        print(f"Fold {fold} (head={head})")
        # --- extract data & prepare ---
        X_tr, y_tr = X_all[tr], y_all[tr]
        df_tr = pd.DataFrame(X_tr, columns=feature_cols)
        df_tr[target_col] = y_tr
        df_tr_clean = iblr.smote(data=df_tr, y=target_col, rel_coef=0.5)
        df_tr_clean = df_tr_clean.dropna() if df_tr_clean.isnull().values.any() else df_tr_clean
        df_tr_clean = df_tr_clean.loc[:, df_tr_clean.nunique() > 1]
        # 3) Try SMOTE with a fallback
        try:
           df_tr_bal = iblr.enn(data=df_tr_clean, y=target_col, rel_coef=0.5)
        except ValueError as e:
           print(f"[Fold {fold}] SMOTE failed ({e}); retrying with rel_coef=0.25")
        try:
           df_tr_bal = iblr.enn(data=df_tr_clean, y=target_col, rel_coef=0.25)
        except ValueError as e2:
           print(f"[Fold {fold}] SMOTE retry failed ({e2}); skipping SMOTE")
        df_tr_bal = df_tr_clean.copy()
        X_tr_bal = df_tr_bal.iloc[:, :-1].values
        y_tr_bal = df_tr_bal[target_col].values

        # Build & train transformer…
        m = build_transformer_model(
            num_features=X_all.shape[1], task='regression', head_type=head,
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            ff_dim=args.ff_dim, num_transformer_blocks=args.num_transformer_blocks,
            mlp_units=args.mlp_units, dropout_rate=args.dropout_rate,
            l2_reg=args.l2_reg
        )
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='log_cosh')
        sc_y = QuantileTransformer()
        y_tr_s = sc_y.fit_transform(y_tr_bal.reshape(-1,1)).flatten()
        y_vl_s = sc_y.transform(y_all[vl].reshape(-1,1)).flatten()
        m.fit(X_tr_bal, y_tr_s,
              validation_data=(X_all[vl], y_vl_s),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                                       patience=args.lr_patience, min_lr=args.min_lr)
              ], verbose=0)
        transformer_models.append((m, sc_y))

        # Validation predictions & metrics
        preds_vl = sc_y.inverse_transform(m.predict(X_all[vl]).flatten().reshape(-1,1)).flatten()
        true_vl = y_all[vl]
        rmse_val = np.sqrt(mean_squared_error(true_vl, preds_vl))
        mae_val  = mean_absolute_error(true_vl, preds_vl)
        mean_true = np.mean(true_vl) or 1.0

        cv_metrics['Transformer']['r2'].append(r2_score(true_vl, preds_vl))
        cv_metrics['Transformer']['rmse'].append(rmse_val)
        cv_metrics['Transformer']['mae'].append(mae_val)
        cv_metrics['Transformer']['nrmse'].append(rmse_val / mean_true)
        cv_metrics['Transformer']['nmae'].append(mae_val  / mean_true)

    # Ensemble‐predict on test set
    preds_s = np.mean([mdl.predict(X_test).flatten() for mdl, _ in transformer_models], axis=0)
    preds   = transformer_models[-1][1].inverse_transform(preds_s.reshape(-1,1)).flatten()
    rmse_test = np.sqrt(mean_squared_error(y_test, preds))
    mae_test  = mean_absolute_error(y_test, preds)
    mean_test = np.mean(y_test) or 1.0

    test_metrics['Transformer']['r2']    = r2_score(y_test, preds)
    test_metrics['Transformer']['rmse']  = rmse_test
    test_metrics['Transformer']['mae']   = mae_test
    test_metrics['Transformer']['nrmse'] = rmse_test / mean_test
    test_metrics['Transformer']['nmae']  = mae_test  / mean_test

    # b) Baselines CV and test
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor

    baseline_classes = {
        'Ridge': Ridge,
        'SVR': SVR,
        'RandomForest': lambda **kw: RandomForestRegressor(random_state=args.seed, **kw)
    }

    for name, Cls in baseline_classes.items():
        fold_models = []
        for fold, (tr, vl) in enumerate(kf.split(X_all, y_all), 1):
            mdl = Cls()
            mdl.fit(X_tr_bal, y_tr_bal)
            fold_models.append(mdl)

            preds_vl = mdl.predict(X_all[vl])
            true_vl  = y_all[vl]
            rmse_val = np.sqrt(mean_squared_error(true_vl, preds_vl))
            mae_val  = mean_absolute_error(true_vl, preds_vl)
            mean_true = np.mean(true_vl) or 1.0

            cv_metrics[name]['r2'].append(r2_score(true_vl, preds_vl))
            cv_metrics[name]['rmse'].append(rmse_val)
            cv_metrics[name]['mae'].append(mae_val)
            cv_metrics[name]['nrmse'].append(rmse_val / mean_true)
            cv_metrics[name]['nmae'].append(mae_val  / mean_true)

        preds_test = np.mean([m.predict(X_test) for m in fold_models], axis=0)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
        mae_test  = mean_absolute_error(y_test, preds_test)
        mean_test = np.mean(y_test) or 1.0

        test_metrics[name]['r2']    = r2_score(y_test, preds_test)
        test_metrics[name]['rmse']  = rmse_test
        test_metrics[name]['mae']   = mae_test
        test_metrics[name]['nrmse'] = rmse_test / mean_test
        test_metrics[name]['nmae']  = mae_test  / mean_test

    # 4) Save all metrics for MATLAB plotting (including normalized ones)
    savemat('clearance_results_cv.mat', {
        # CV folds arrays
        'cv_r2_transformer':    np.array(cv_metrics['Transformer']['r2']),
        'cv_rmse_transformer':  np.array(cv_metrics['Transformer']['rmse']),
        'cv_mae_transformer':   np.array(cv_metrics['Transformer']['mae']),
        'cv_nrmse_transformer': np.array(cv_metrics['Transformer']['nrmse']),
        'cv_nmae_transformer':  np.array(cv_metrics['Transformer']['nmae']),
        # … and same for Ridge, SVR, RF …
        'cv_r2_ridge':          np.array(cv_metrics['Ridge']['r2']),
        'cv_rmse_ridge':        np.array(cv_metrics['Ridge']['rmse']),
        'cv_mae_ridge':         np.array(cv_metrics['Ridge']['mae']),
        'cv_nrmse_ridge':       np.array(cv_metrics['Ridge']['nrmse']),
        'cv_nmae_ridge':        np.array(cv_metrics['Ridge']['nmae']),
        'cv_r2_svr':            np.array(cv_metrics['SVR']['r2']),
        'cv_rmse_svr':          np.array(cv_metrics['SVR']['rmse']),
        'cv_mae_svr':           np.array(cv_metrics['SVR']['mae']),
        'cv_nrmse_svr':         np.array(cv_metrics['SVR']['nrmse']),
        'cv_nmae_svr':          np.array(cv_metrics['SVR']['nmae']),
        'cv_r2_rf':             np.array(cv_metrics['RandomForest']['r2']),
        'cv_rmse_rf':           np.array(cv_metrics['RandomForest']['rmse']),
        'cv_mae_rf':            np.array(cv_metrics['RandomForest']['mae']),
        'cv_nrmse_rf':          np.array(cv_metrics['RandomForest']['nrmse']),
        'cv_nmae_rf':           np.array(cv_metrics['RandomForest']['nmae']),
        # Test‐set arrays (Transformer, Ridge, SVR, RF)
        'test_r2_all':    np.array([test_metrics[m]['r2']   for m in ['Transformer','Ridge','SVR','RandomForest']]),
        'test_rmse_all':  np.array([test_metrics[m]['rmse'] for m in ['Transformer','Ridge','SVR','RandomForest']]),
        'test_mae_all':   np.array([test_metrics[m]['mae']  for m in ['Transformer','Ridge','SVR','RandomForest']]),
        'test_nrmse_all': np.array([test_metrics[m]['nrmse'] for m in ['Transformer','Ridge','SVR','RandomForest']]),
        'test_nmae_all':  np.array([test_metrics[m]['nmae'] for m in ['Transformer','Ridge','SVR','RandomForest']]),
    })
    print("✔ Saved CV folds, test metrics, and normalized metrics to 'clearance_results_cv.mat'")

# ------------------------------------------------------------------------------
# 5) Argparse & Main
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['regression','classification'], required=True)
    p.add_argument('--train_file', required=True, help='Path to training CSV file')
    p.add_argument('--test_file',  required=True, help='Path to test CSV file')
    p.add_argument('--head_type', choices=['mlp','rbf','poly','kan','interaction','all'], default='mlp')
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
    p.add_argument('--patience', type=int, default=20)
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
