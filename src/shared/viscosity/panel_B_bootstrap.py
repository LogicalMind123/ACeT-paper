
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
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from viscosity.model import DenseKANRBF, build_transformer_model

# ------------------------------------------------------------------------------
# 3) Regression Pipeline
# ------------------------------------------------------------------------------
def run_regression(args):
    print("=== Running Regression Task with Bootstrap Stratified Analysis ===")
    train = pd.read_csv(args.train_file)
    test = pd.read_csv(args.test_file)
    target_col = train.columns[-1]

    # Synthetic data generation and combination
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
    synthesizer.fit(train)
    synthetic_data = synthesizer.sample(num_rows=26)
    train_mix = pd.concat([train, train, synthetic_data], ignore_index=True)

    # Scaling
    sc_X = QuantileTransformer()
    X_all = sc_X.fit_transform(train_mix.iloc[:, :-1].values)
    X_test = sc_X.transform(test.iloc[:, :-1].values)
    y_all = train_mix.iloc[:, -1].values
    feature_cols = train_mix.columns[:-1]

    heads = [args.head_type] if args.head_type != 'all' else ['mlp','rbf','spline','kan','interaction']
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for head in heads:
        ens_models = []
        for fold, (tr, vl) in enumerate(kf.split(X_all, y_all), 1):
            X_tr, y_tr = X_all[tr], y_all[tr]

            df_tr = pd.DataFrame(X_tr, columns=feature_cols)
            df_tr[target_col] = y_tr

            df_tr_clean = iblr.enn(data=df_tr, y=target_col, rel_coef=0.5)
            df_tr_bal = iblr.smote(data=df_tr_clean, y=target_col, rel_coef=0.5)

            X_tr_bal = df_tr_bal.iloc[:, :-1].values
            y_tr_bal = df_tr_bal[target_col].values

            model = build_transformer_model(
                num_features=X_all.shape[1], task='regression', head_type=head,
                embed_dim=args.embed_dim, num_heads=args.num_heads,
                ff_dim=args.ff_dim, num_transformer_blocks=args.num_transformer_blocks,
                mlp_units=args.mlp_units, dropout_rate=args.dropout_rate, l2_reg=args.l2_reg
            )

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='log_cosh')

            sc_y = QuantileTransformer()
            y_tr_s = sc_y.fit_transform(y_tr_bal.reshape(-1, 1)).flatten()

            model.fit(
                X_tr_bal, y_tr_s,
                validation_data=(X_all[vl], sc_y.transform(y_all[vl].reshape(-1, 1)).flatten()),
                epochs=args.epochs, batch_size=args.batch_size,
                callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)],
                verbose=0
            )

            ens_models.append((model, sc_y))

        # Final ensemble predictions
        preds_s = np.mean([mdl.predict(X_test).flatten() for mdl, _ in ens_models], axis=0)
        preds = ens_models[-1][1].inverse_transform(preds_s.reshape(-1, 1)).flatten()
        trues = test.iloc[:, -1].values

        # Bootstrap stratified analysis
        df_test = pd.DataFrame({'True': trues, 'Pred': preds})
        bins = np.quantile(trues, [0, 0.25, 0.75, 1.0])
        labels = ['Low', 'Mid', 'High']
        df_test['Bin'] = pd.cut(df_test['True'], bins=bins, labels=labels, include_lowest=True)

        bootstrap_iterations = 1000
        bootstrap_metrics = []
        all_rmse_samples = {}
        for label in labels:
            bin_data = df_test[df_test['Bin'] == label]
            if len(bin_data) < 3:
                continue
            rmse_samples, r2_samples, mae_samples = [], [], []
            for _ in range(bootstrap_iterations):
                sample = bin_data.sample(frac=1, replace=True)
                rmse_samples.append(np.sqrt(mean_squared_error(sample['True'], sample['Pred'])))
                r2_samples.append(r2_score(sample['True'], sample['Pred']))
                mae_samples.append(mean_absolute_error(sample['True'], sample['Pred']))
            mat_label = f'rmse_{label.lower()}'
            all_rmse_samples[mat_label] = np.array(rmse_samples)
            bootstrap_metrics.append({
                'Bin': label,
                'RMSE_mean': np.mean(rmse_samples),
                'RMSE_std': np.std(rmse_samples),
                'R2_mean': np.mean(r2_samples),
                'R2_std': np.std(r2_samples),
                'MAE_mean': np.mean(mae_samples),
                'MAE_std': np.std(mae_samples),
                'all_rmse_samples': all_rmse_samples
            })

        bootstrap_df = pd.DataFrame(bootstrap_metrics)
        bootstrap_df.to_csv('viscosity_bootstrap_metrics.csv', index=False)
        print("Bootstrap metrics saved explicitly as 'viscosity_bootstrap_metrics.csv'.")

        # Convert dataframe to dict explicitly suitable for MATLAB
        matlab_vars = {
           'Bin': bootstrap_df['Bin'].values.astype('object'),
           'RMSE_mean': bootstrap_df['RMSE_mean'].values,
           'RMSE_std': bootstrap_df['RMSE_std'].values,
           'R2_mean': bootstrap_df['R2_mean'].values,
           'R2_std': bootstrap_df['R2_std'].values,
           'MAE_mean': bootstrap_df['MAE_mean'].values,
           'MAE_std': bootstrap_df['MAE_std'].values,
           'all_rmse_samples': all_rmse_samples

        }

        # Explicitly save as .mat file
        savemat('viscosity_bootstrap_metrics.mat', matlab_vars)

        print("Bootstrap metrics saved explicitly as 'viscosity_bootstrap_metrics.csv' and 'viscosity_bootstrap_metrics.mat'.")


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
