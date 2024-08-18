import os
import random
import sys
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed
import optuna

# Custom functions
from funcs.api_funcs import (
    get_feature_addition_rounds_arg, 
    get_target_arg, 
    get_feature_dropping_threshold_arg, 
    get_tsfresh_fc_params_arg
)
from funcs.engineer_features_funcs import (
    compute_mse_scores, 
    optimize_params, 
    evaluate_feature,
    extract_tsfresh_features, 
    compute_baseline_mse, 
    compute_mse_with_added_feature, 
    compute_mse_with_dropped_feature,  
)
from funcs.dvc_funcs import get_repo_bucket_client, dagshub_initialization

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main(target, feature_addition_rounds, feature_dropping_threshold, fc_parameters, X_train, X_test, y_train, y_test):
    # Verify and initialize DVC remote configuration
    logger.debug("Calling dagshub_initialization()")
    dagshub_initialization()
    logger.debug("dagshub_initialization() called successfully")
    
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # Combine X and y dataframes for feature engineering
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    # Ensure the column names are preserved
    base_features = list(X_train.columns)

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Initial baseline MSE calculation
    logger.debug("Calculating initial baseline MSE")
    baseline_mse_scores, aggregated_baseline_mse, best_params = compute_mse_scores(
        X_train, X_test, y_train, y_test, base_features
    )

    logger.debug(f"Baseline MSE Scores: {baseline_mse_scores}")
    logger.debug(f"Aggregated Baseline MSE Score: {aggregated_baseline_mse}")

    initial_baseline_mse = aggregated_baseline_mse
    initial_mse_xgboost = baseline_mse_scores['XGBoost']
    initial_mse_lightgbm = baseline_mse_scores['LightGBM']

    # Scale the data
    logger.debug("Scaling the data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.debug("Data scaled successfully")

    # Optimize parameters
    logger.debug("Optimizing XGBoost parameters")
    xgboost_params = optimize_params('XGBoost', X_train_scaled, y_train, X_test_scaled, y_test, n_trials=10)

    logger.debug("Optimizing LightGBM parameters")
    lightgbm_params = optimize_params('LightGBM', X_train_scaled, y_train, X_test_scaled, y_test, n_trials=10)

    # Save as dictionaries to a Python file
    logger.debug("Saving best parameters to a Python file")
    with open('best_params.py', 'w') as f:
        f.write(f"xgboost_params = {xgboost_params}\n")
        f.write(f"lightgbm_params = {lightgbm_params}\n")

    logger.debug("Best parameters saved to best_params.py")

    # Skip TSFRESH feature engineering if feature_addition_rounds is 0
    if feature_addition_rounds > 0:
        # Add ID and time columns required by TSFRESH
        train_combined['id'] = 1
        train_combined['time'] = train_combined.index

        test_combined['id'] = 2
        test_combined['time'] = test_combined.index

        # Extract TSFRESH features once
        logger.debug("Extracting TSFRESH features for train set")
        tsfresh_features_train = extract_tsfresh_features(train_combined, column_id='id', column_sort='time', default_fc_parameters=fc_parameters)
        tsfresh_features_train.index = train_combined.index[:len(tsfresh_features_train)]
        logger.debug("TSFRESH features extracted for train set")

        logger.debug("Extracting TSFRESH features for test set")
        tsfresh_features_test = extract_tsfresh_features(test_combined, column_id='id', column_sort='time', default_fc_parameters=fc_parameters)
        tsfresh_features_test.index = test_combined.index[:len(tsfresh_features_test)]
        logger.debug("TSFRESH features extracted for test set")

        all_added_features = []

        # Initialize the baseline MSE without adding any new feature
        baseline_mse_scores, aggregated_baseline_mse = compute_baseline_mse(
            X_train, X_test, y_train, y_test, base_features
        )
        initial_baseline_mse = aggregated_baseline_mse

        # TSFRESH feature selection rounds with parallel processing
        for round_num in range(feature_addition_rounds):
            logger.debug(f"Round {round_num + 1}/{feature_addition_rounds} of feature engineering")

            # Evaluate adding features in parallel
            results = Parallel(n_jobs=-1)(delayed(evaluate_feature)(
                feature, tsfresh_features_train, tsfresh_features_test, X_train, X_test,
                y_train, y_test, base_features, aggregated_baseline_mse, all_added_features
            ) for feature in tsfresh_features_train.columns)
            aggregated_mse_scores_added = [res for res in results if res is not None]

            # Sort and add top 3 features if they improve the model
            aggregated_mse_scores_added.sort(key=lambda x: x[1])
            top_three_to_add = [f for f in aggregated_mse_scores_added[:3] if f[2] > 0]

            if not top_three_to_add:
                logger.debug("No features improved the model in this round.")
                continue

            for feature, _, improvement, _, _ in top_three_to_add:
                improvement = float(improvement)
                base_features.append(feature)
                all_added_features.append(feature)
                X_train[feature] = tsfresh_features_train[feature]
                X_test[feature] = tsfresh_features_test[feature]
                logger.debug(f"Feature added: {feature}, Improvement: {improvement}")

                # Update baseline MSE after adding each feature
                baseline_mse_scores, aggregated_baseline_mse = compute_mse_with_added_feature(
                    X_train, X_test, y_train, y_test, base_features, feature
                )

        # Calculate overall improvement
        overall_improvement = initial_baseline_mse - aggregated_baseline_mse
        logger.debug(f"\nOverall Improvement in MSE after {feature_addition_rounds} rounds: {overall_improvement}")

        # Use base_features and all_added_features to filter out features that weren't picked
        all_features = list(set(base_features + all_added_features))
        X_train = X_train[all_features]
        X_test = X_test[all_features]

        logger.debug("Feature Addition Complete")

    # Set a higher threshold for improvement (e.g., 0.05% of the initial baseline MSE)
    feature_dropping_threshold = float(feature_dropping_threshold)
    threshold = feature_dropping_threshold * ((baseline_mse_scores['XGBoost'] + baseline_mse_scores['LightGBM']) / 2)

    # Calculate and drop features
    aggregated_mse_scores_dropped = []
    for feature in base_features:
        mse_scores, aggregated_mse = compute_mse_with_dropped_feature(X_train, X_test, y_train, y_test, base_features, feature)
        improvement = aggregated_baseline_mse - aggregated_mse

        improvement_status = "improved" if improvement > threshold else "worsened"
        aggregated_mse_scores_dropped.append((feature, aggregated_mse, improvement, improvement_status, mse_scores))

    # Sort and drop the least impactful features if they result in improvement
    aggregated_mse_scores_dropped.sort(key=lambda x: x[1])
    features_to_drop = [f for f in aggregated_mse_scores_dropped if f[2] > threshold]

    if not features_to_drop:
        print("No features were dropped as they did not improve the model.")
    else:
        for feature, _, improvement, _, _ in features_to_drop:
            base_features.remove(feature)
            print(f"Feature dropped: {feature}, Improvement: {improvement}")

    print("Feature Dropping Completed.")

    # Final baseline MSE calculation
    final_mse_scores, aggregated_final_mse, _ = compute_mse_scores(
        X_train, X_test, y_train, y_test, base_features
    )
    # Store final MSE scores for each model
    final_mse_xgboost = final_mse_scores['XGBoost']
    final_mse_lightgbm = final_mse_scores['LightGBM']

    # Calculate improvements
    improvement_xgboost = initial_mse_xgboost - final_mse_xgboost
    improvement_lightgbm = initial_mse_lightgbm - final_mse_lightgbm

    # Print the results
    print("")
    print("THRESHOLD OF 0.002:")
    print(f"Initial MSE for XGBoost: {initial_mse_xgboost}")
    print(f"Final MSE for XGBoost: {final_mse_xgboost}")
    print(f"Improvement in MSE for XGBoost: {improvement_xgboost}\n")

    print(f"Initial MSE for LightGBM: {initial_mse_lightgbm}")
    print(f"Final MSE for LightGBM: {final_mse_lightgbm}")
    print(f"Improvement in MSE for LightGBM: {improvement_lightgbm}")

    # Print the results
    logger.debug(f"Initial MSE for XGBoost: {initial_mse_xgboost}")
    logger.debug(f"Final MSE for XGBoost: {final_mse_xgboost}")
    logger.debug(f"Improvement in MSE for XGBoost: {improvement_xgboost}\n")

    logger.debug(f"Initial MSE for LightGBM: {initial_mse_lightgbm}")
    logger.debug(f"Final MSE for LightGBM: {final_mse_lightgbm}")
    logger.debug(f"Improvement in MSE for LightGBM: {improvement_lightgbm}")

    # Save Data using S3 buckets and .csv files
    if not os.path.exists('data/engineered'):
        os.makedirs('data/engineered')
    X_train.to_csv('data/engineered/X_train_engineered.csv', index=True)
    X_test.to_csv('data/engineered/X_test_engineered.csv', index=True)
    y_train.to_csv('data/engineered/y_train_engineered.csv', index=True)
    y_test.to_csv('data/engineered/y_test_engineered.csv', index=True)

    logger.debug("Engineered data saved locally.")

    # Upload to Dagshub storage
    logger.debug("Uploading to DAGsHub storage")
    s3 = get_repo_bucket_client("najibabounasr/MacroEconomicAPI")
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/engineered/X_train_engineered.csv",
        Key="data/engineered/X_train_engineered.csv",
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/engineered/X_test_engineered.csv",
        Key="data/engineered/X_test_engineered.csv",
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/engineered/y_train_engineered.csv",
        Key="data/engineered/y_train_engineered.csv",
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/engineered/y_test_engineered.csv",
        Key="data/engineered/y_test_engineered.csv",
    )
    logger.debug("Data uploaded to DAGsHub storage")

if __name__ == '__main__':
    target = get_target_arg()
    feature_addition_rounds = get_feature_addition_rounds_arg()
    feature_dropping_threshold = float(get_feature_dropping_threshold_arg())
    tsfresh_fc_params = str(get_tsfresh_fc_params_arg())
    X_train = pd.read_csv('data/processed/X_train_transformed.csv', index_col='Date', parse_dates=True)
    y_train = pd.read_csv('data/processed/y_train_transformed.csv', index_col='Date', parse_dates=True)
    y_test = pd.read_csv('data/processed/y_test_transformed.csv', index_col='Date', parse_dates=True)
    X_test = pd.read_csv('data/processed/X_test_transformed.csv', index_col='Date', parse_dates=True)

    # Map the string to the appropriate TSFRESH parameter object
    if tsfresh_fc_params == 'MinimalFCParameters':
        fc_parameters = MinimalFCParameters()
    elif tsfresh_fc_params == 'ComprehensiveFCParameters':
        fc_parameters = ComprehensiveFCParameters()
    elif tsfresh_fc_params == 'EfficientFCParameters':
        fc_parameters = EfficientFCParameters()
    else:
        # Default to MinimalFCParameters if the provided value is invalid
        fc_parameters = MinimalFCParameters()
        logger.debug("Invalid TSFRESH feature extraction parameters. Defaulting to 'MinimalFCParameters'.")
        print("Invalid TSFRESH feature extraction parameters. Choose from: 'MinimalFCParameters','EfficientFCParameters','ComprehensiveFCParameters'.")

    logger.debug("Starting feature engineering")
    main(target, feature_addition_rounds, feature_dropping_threshold, fc_parameters, X_train, X_test, y_train, y_test)
    logger.debug("Feature engineering finished")
