import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed
import optuna
from best_params import xgboost_params, lightgbm_params
import re
import ast


# Function to optimize parameters using Optuna
def optimize_params(model_name, X_train, y_train, X_test, y_test, n_trials):
    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
            }
            model = XGBRegressor(**params)
        elif model_name == 'LightGBM':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
            }
            model = LGBMRegressor(**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials, n_jobs=1)
    return study.best_params

# Function to compute MSE scores
def compute_mse_scores(X_train, X_test, y_train, y_test, features):
    if not features:  # If no features are provided, return a high MSE (or a default value)
        return {'XGBoost': np.inf, 'LightGBM': np.inf}, np.inf, {}

    X_train = X_train[features].dropna().values
    X_test = X_test[features].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params),
        'LightGBM': LGBMRegressor(**lightgbm_params)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    return mse_scores, aggregated_mse, {}

# Function to evaluate a single feature addition
def evaluate_feature(feature, tsfresh_features_train, tsfresh_features_test, X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, base_features, aggregated_baseline_mse, all_added_features):
    if feature in all_added_features:
        return None
    if feature not in tsfresh_features_train.columns or feature not in tsfresh_features_test.columns:
        return None

    temp_X_train = pd.concat([X_train_transformed, tsfresh_features_train[[feature]]], axis=1)
    temp_X_test = pd.concat([X_test_transformed, tsfresh_features_test[[feature]]], axis=1)
    mse_scores, aggregated_mse = compute_mse_with_added_feature(temp_X_train, temp_X_test, y_train_transformed, y_test_transformed, base_features, feature)

    improvement = aggregated_baseline_mse - aggregated_mse

    improvement_status = "improved" if improvement > 0 else "worsened"
    return (feature, aggregated_mse, improvement, improvement_status, mse_scores)

# Function to compute MSE scores after adding a feature
def compute_mse_with_added_feature(X_train, X_test, y_train, y_test, base_features, add_feature):
    X_train = X_train[base_features + [add_feature]].dropna().values
    X_test = X_test[base_features + [add_feature]].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params),
        'LightGBM': LGBMRegressor(**lightgbm_params)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    return mse_scores, aggregated_mse

# Function to compute MSE scores after dropping a feature
def compute_mse_with_dropped_feature(X_train, X_test, y_train, y_test, base_features, drop_feature, aggregated_baseline_mse):
    remaining_features = [f for f in base_features if f != drop_feature]
    X_train_dropped = X_train[remaining_features].dropna().values
    X_test_dropped = X_test[remaining_features].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params),
        'LightGBM': LGBMRegressor(**lightgbm_params)
    }

    for model_name, model in models.items():
        model.fit(X_train_dropped, y_train)
        y_pred = model.predict(X_test_dropped)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    improvement = aggregated_baseline_mse - aggregated_mse
    improvement_status = "improved" if improvement > 0 else "worsened"
    return (drop_feature, aggregated_mse, improvement, mse_scores, improvement_status)

# Paths for processed data and TSFRESH features
processed_train_path = 'data/processed/train_transformed_combined.csv'
processed_test_path = 'data/processed/test_transformed_combined.csv'
tsfresh_train_path = 'data/tsfresh/train_combined_all_features_filled.csv'
tsfresh_test_path = 'data/tsfresh/test_combined_all_features_filled.csv'

# Load data
train_combined = pd.read_csv(processed_train_path, index_col='Date', parse_dates=True)
test_combined = pd.read_csv(processed_test_path, index_col='Date', parse_dates=True)
tsfresh_features_train = pd.read_csv(tsfresh_train_path, index_col='Date', parse_dates=True)
tsfresh_features_test = pd.read_csv(tsfresh_test_path, index_col='Date', parse_dates=True)

# Path to results file
feature_engineering_results = pd.read_csv(r'C:\Users\nabounaser\OneDrive - Ejada Systems\EJADA\MacroEconomicAPI\feature_engineering_results.csv')

# Define the base features
base_features = []

# Targets to evaluate
targets = ['FEDFUNDS', 'GDP', 'CPIAUCSL', 'CUSR0000SAH1', 'CPILFESL', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'HOUST', 'DSPI', 
           'DGS2', 'DGS5', 'DGS10', 'AAA', 'BAA', 'WTISPLC', 'IMPGS', 'GCE', 'FGCE', 'GDPCTPI', 'PCEPI', 'PCEPILFE', 
           'PAYEMS', 'UNRATE', 'INDPRO', 'CUMFNS', 'USREC']

# Model parameters
xgboost_params = {'max_depth': 9, 'learning_rate': 0.0168, 'n_estimators': 193, 'min_child_weight': 3, 'subsample': 0.5866, 'colsample_bytree': 0.9177, 'reg_alpha': 0.0125, 'reg_lambda': 8.577e-05, 'verbosity': 0}
lightgbm_params = {'num_leaves': 119, 'learning_rate': 0.0832, 'n_estimators': 71, 'max_depth': 3, 'min_child_samples': 18, 'subsample': 0.7402, 'colsample_bytree': 0.9405, 'reg_alpha': 0.000756, 'reg_lambda': 0.000259, 'verbosity': -1}

# Relative threshold for feature dropping
relative_threshold = 0.000002

# Loop through each target
results = []

for target in targets:
    print(f"\nProcessing target: {target}")
    
    # Extract base features for the target
    base_features_str = feature_engineering_results.loc[feature_engineering_results['target'] == target]['final_feature_space'].values[0]
    base_features = ast.literal_eval(base_features_str)

    # Clean the double double-quotes
    base_features = [re.sub(r'""(.*?)""', r'"\1"', feature) for feature in base_features]

    # Verify that these features exist in the tsfresh data
    valid_features = [feature for feature in base_features if feature in tsfresh_features_train.columns]

    X_train = tsfresh_features_train[valid_features]  # Start with valid base features
    X_test = tsfresh_features_test[valid_features]  # Start with valid base features
    y_train = train_combined[[target]]
    y_test = test_combined[[target]]
    
    all_added_features = valid_features.copy()

    # Initial baseline MSE calculation
    baseline_mse_scores, aggregated_baseline_mse, _ = compute_mse_scores(X_train, X_test, y_train, y_test, valid_features)
    print(f"Initial aggregated baseline MSE: {aggregated_baseline_mse}")
    print(f"Initial MSE for XGBoost: {baseline_mse_scores['XGBoost']}")
    print(f"Initial MSE for LightGBM: {baseline_mse_scores['LightGBM']}")

    for round_num in range(4):
        print(f"\n---- Round {round_num + 1} of feature addition ----")

        # Evaluate adding features
        results_added = Parallel(n_jobs=-1)(delayed(evaluate_feature)(
            feature, tsfresh_features_train, tsfresh_features_test, X_train, X_test,
            y_train, y_test, all_added_features, aggregated_baseline_mse, all_added_features
        ) for feature in tsfresh_features_train.columns)
        
        results_added = [res for res in results_added if res is not None]
        results_added.sort(key=lambda x: x[1])

        if results_added:
            print(f"Top features considered for addition: {[f[0] for f in results_added[:3]]}")
        
        top_to_add = [f for f in results_added[:3] if f[2] > 0]
        for feature, _, improvement, _, _ in top_to_add:
            all_added_features.append(feature)
            X_train[feature] = tsfresh_features_train[feature]
            X_test[feature] = tsfresh_features_test[feature]
            print(f"Added feature: {feature} with improvement: {improvement}")

        # Recalculate the baseline MSE after feature addition
        baseline_mse_scores, aggregated_baseline_mse, _ = compute_mse_scores(X_train, X_test, y_train, y_test, all_added_features)
        print(f"New aggregated MSE after addition: {aggregated_baseline_mse}")
        print(f"New MSE for XGBoost: {baseline_mse_scores['XGBoost']}")
        print(f"New MSE for LightGBM: {baseline_mse_scores['LightGBM']}")

        # Calculate the dynamic threshold for this round
        threshold = relative_threshold * aggregated_baseline_mse

        print(f"\n---- Round {round_num + 1} of feature dropping ----")

        # Evaluate dropping features
        results_dropped = Parallel(n_jobs=-1)(delayed(compute_mse_with_dropped_feature)(
            X_train, X_test, y_train, y_test, all_added_features, feature, aggregated_baseline_mse
        ) for feature in all_added_features)

        results_dropped.sort(key=lambda x: x[1])

        if results_dropped:
            print(f"Features considered for dropping (with their improvements): {[f[0] for f in results_dropped]}")

        top_to_drop = [f for f in results_dropped if f[2] > threshold]

        for feature, _, improvement, _, _ in top_to_drop:
            if improvement > threshold:
                all_added_features.remove(feature)
                X_train.drop(columns=[feature], inplace=True)
                X_test.drop(columns=[feature], inplace=True)
                print(f"Dropped feature: {feature} with improvement: {improvement}")

        # Recalculate the baseline MSE after feature dropping
        final_mse_scores, final_aggregated_mse, _ = compute_mse_scores(X_train, X_test, y_train, y_test, all_added_features)
        print(f"Final aggregated MSE after dropping: {final_aggregated_mse}")
        print(f"Final MSE for XGBoost: {final_mse_scores['XGBoost']}")
        print(f"Final MSE for LightGBM: {final_mse_scores['LightGBM']}")

    results.append({
        'target': target,
        'initial_mse_xgboost': baseline_mse_scores['XGBoost'],
        'initial_mse_lightgbm': baseline_mse_scores['LightGBM'],
        'final_mse_xgboost': final_mse_scores['XGBoost'],
        'final_mse_lightgbm': final_mse_scores['LightGBM'],
        'initial_aggregated_mse': aggregated_baseline_mse,
        'final_aggregated_mse': final_aggregated_mse,
        'improvement': aggregated_baseline_mse - final_aggregated_mse,
        'final_feature_space': all_added_features
    })

    # Save the final feature list for the target
    features_df = pd.DataFrame({'features': all_added_features})
    features_df.to_csv(f'{target}_final_features.csv', index=False)

# Save the overall results
results_df = pd.DataFrame(results)
results_df.to_csv('feature_engineering_results_round2.csv', index=False)

print("Feature engineering completed for all targets.")
# turn off all warnings
warnings.filterwarnings('ignore')
# FORCE ALLLLL WARNINGS TO STOP IMMEDIEATELY
warnings.filterwarnings(action='ignore')
