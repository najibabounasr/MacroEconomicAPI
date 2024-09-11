import os
import pandas as pd
import numpy as np
import mlflow
import dagshub
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from funcs.api_funcs import get_target_arg, get_feature_addition_rounds_arg, get_feature_dropping_threshold_arg, get_tsfresh_fc_params_arg
from sklearn.neighbors import KNeighborsRegressor
from autogluon.tabular import TabularPredictor
from funcs.train_model_funcs import clean_data

def optimize_model(objective_function, model_name):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_function, n_trials=100)
    return study

# Define objective functions for each model
def objective_knn(trial, X_train, y_train, X_test, y_test, target_feature):
    param_grid = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 50),
        'p': trial.suggest_int('p', 1, 2)
    }
    
    with mlflow.start_run(nested=True):
        model = KNeighborsRegressor(**param_grid)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        mlflow.log_metrics({'test_rmse': test_rmse})
        
        for param_key, param_value in param_grid.items():
            mlflow.log_param(param_key, param_value)
        
        mlflow.set_tags({
            'target_feature': target_feature,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters),
            'version' : '1.0.0'
        })

    return test_rmse

def objective_autogluon(trial, X_train, y_train, X_test, y_test, target):
    param_grid = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'num_boost_round': trial.suggest_int('num_boost_round', 50, 100),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-4, 1e+1),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1e+1),
        'verbose': -1
    }
    
    with mlflow.start_run(nested=True):
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.columns = list(X_train.columns) + [target]
        
        predictor = TabularPredictor(label=target, eval_metric='rmse').fit(
            train_data=train_data,
            hyperparameters={'GBM': param_grid},
            num_bag_folds=5,
            ag_args_fit={'num_gpus': 0, 'num_cpus': 1}
        )
        test_predictions = predictor.predict(X_test)
        train_predictions = predictor.predict(X_train)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

        mlflow.log_metrics({
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        })
        
        for param_key, param_value in param_grid.items():
            mlflow.log_param(param_key, param_value)
        
        mlflow.set_tags({
            'target_feature': target_feature,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters),
            'version' : '1.0.0'
        })

    return test_rmse

def objective_lgb(trial, X_train, y_train, X_test, y_test, target_feature):
    param_grid = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'verbosity': -1
    }
    
    with mlflow.start_run(nested=True):
        model = lgb.LGBMRegressor(**param_grid)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        train_predictions = model.predict(X_train)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

        mlflow.log_metrics({
            'test_rmse': test_rmse,
            'train_rmse': train_rmse
        })
        
        for param_key, param_value in param_grid.items():
            mlflow.log_param(param_key, param_value)
        
        mlflow.set_tags({
            'target_feature': target_feature,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters),
            'version' : '1.0.0'
        })

    return test_rmse

def objective_xgb(trial, X_train, y_train, X_test, y_test, target_feature):
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-4, 1e+1),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e+1),
        'verbosity': 0
    }
    
    with mlflow.start_run(nested=True):
        model = xgb.XGBRegressor(**param_grid)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

        mlflow.log_metrics({'test_rmse': test_rmse})
        
        for param_key, param_value in param_grid.items():
            mlflow.log_param(param_key, param_value)
        
        mlflow.set_tags({
            'target_feature': target_feature,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters),
            'version' : '1.0.0'
        })

    return test_rmse

def objective_catboost(trial, X_train, y_train, X_test, y_test, target_feature):
    param_grid = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 1e+1),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_uniform('random_strength', 0.0, 1.0),
        'verbose': 0
    }
    
    with mlflow.start_run(nested=True):
        model = cb.CatBoostRegressor(**param_grid, verbose=0)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        train_predictions = model.predict(X_train)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))

        mlflow.log_metrics({
            'test_rmse': test_rmse,
            'train_rmse': train_rmse
        })
        
        for param_key, param_value in param_grid.items():
            mlflow.log_param(param_key, param_value)
        
        mlflow.set_tags({
            'target_feature': target_feature,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters),
            'version' : '1.0.0'
        })

    return test_rmse

def optimize_model(objective_function):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_function, n_trials=100)
    return study

def optimize_all_targets(target_features, X_train, y_train, X_test, y_test):
    for target in target_features:
        mlflow.set_experiment(f"{target} Optimization")
        print(f"Optimizing for target: {target}")
        # Optimize each model for the current target
        optimize_model(lambda trial: objective_knn(trial, X_train, y_train, X_test, y_test, target))
        optimize_model(lambda trial: objective_autogluon(trial, X_train, y_train, X_test, y_test, target))
        optimize_model(lambda trial: objective_lgb(trial, X_train, y_train, X_test, y_test, target))
        optimize_model(lambda trial: objective_xgb(trial, X_train, y_train, X_test, y_test, target))
        optimize_model(lambda trial: objective_catboost(trial, X_train, y_train, X_test, y_test, target))


def main(target_features):
    for target in targets:
        mlflow.set_tracking_uri("https://dagshub.com/najibabounasr/MacroEconomicAPI.mlflow")
        dagshub.init("MacroEconomicAPI", "najibabounasr", mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'najibabounasr'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'fbaccfb8cf4e8d2d195cd05e9a53dbfe32323695'
        # Fix the Base DataFrame
        feature_space = pd.read_csv('feature_engineering_results_round2.csv')
        feature_space_mask = (feature_space == target).any(axis=1)
        feature_space_list = feature_space[feature_space_mask]['final_feature_space'].values.tolist()
        feature_space_series = pd.Series(feature_space_list)
        feature_space = feature_space_series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        flat_feature_space = [item for sublist in feature_space.tolist() for item in sublist] if any(isinstance(i, list) for i in feature_space.tolist()) else feature_space.tolist()
        train_combined_all_features_filed_tsfresh = pd.read_csv('data/tsfresh/train_combined_all_features_filled.csv', index_col='Date', parse_dates=True)
        test_combined_all_features_filed_tsfresh = pd.read_csv('data/tsfresh/test_combined_all_features_filled.csv', index_col='Date', parse_dates=True)
        X_train_tsfresh = train_combined_all_features_filed_tsfresh[flat_feature_space]
        X_test_tsfresh = test_combined_all_features_filed_tsfresh[flat_feature_space]
        X_train_tsfresh = X_train_tsfresh.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        X_test_tsfresh = X_test_tsfresh.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # display(X_train_tsfresh.columns)
        # Fix the Lagged DataFrame
        top_lagged_features = pd.read_csv(r'C:\Users\nabounaser\OneDrive - Ejada Systems\EJADA\MacroEconomicAPI\top_features_combined_rmse_mse_results.csv')

        top_lagged_features_list = top_lagged_features.loc[top_lagged_features['target'] == target]['feature'].tolist()
        top_lagged_features = top_lagged_features[top_lagged_features['target'] == target]
        best_lagged_feature = top_lagged_features.sort_values(by='pct_improvement_mse', ascending=False).iloc[0]['feature']
        combined_lag_train = pd.read_csv('data/engineered/combined_lag_train.csv', index_col='Date', parse_dates=True)
        combined_lag_test = pd.read_csv('data/engineered/combined_lag_test.csv', index_col='Date', parse_dates=True)
        X_train_lagged = combined_lag_train[[best_lagged_feature]]
        X_test_lagged = combined_lag_test[[best_lagged_feature]]
        # display(X_train_lagged.columns)

        # Fix the Base DataFrame
        train_combined = pd.read_csv('data/processed/train_transformed_combined.csv', index_col='Date', parse_dates=True)
        test_combined = pd.read_csv('data/processed/test_transformed_combined.csv', index_col='Date', parse_dates=True)
        X_train = train_combined.drop(columns=[target])
        y_train = train_combined[target]
        X_test = test_combined.drop(columns=[target])
        y_test = test_combined[target]

        # Combine the three DataFrames
        X_train = pd.concat([X_train, X_train_tsfresh, X_train_lagged], axis=1)
        X_test = pd.concat([X_test, X_test_tsfresh, X_test_lagged], axis=1)
        # make sure X_train and X_test have the same columns
        X_train = X_train[X_test.columns]
        # make sure X_train and y_train have the same number of samples/rows
        X_train = X_train.loc[y_train.index]
        # make sure X_test and y_test have the same number of samples/rows
        X_test = X_test.loc[y_test.index]



        mlflow.set_experiment("{target} Optimization")

        print(f"Optimizing KNN...")
        optimize_model(lambda trial: objective_knn(trial, X_train, y_train, X_test, y_test), "KNN")

        print(f"Optimizing AutoGluon...")
        optimize_model(lambda trial: objective_autogluon(trial, X_train, y_train, X_test, y_test, target_feature), "AutoGluon")

        print(f"Optimizing LightGBM...")
        optimize_model(lambda trial: objective_lgb(trial, X_train, y_train, X_test, y_test), "LightGBM")

        print(f"Optimizing XGBoost...")
        optimize_model(lambda trial: objective_xgb(trial, X_train, y_train, X_test, y_test), "XGBoost")

        print(f"Optimizing CatBoost...")
        optimize_model(lambda trial: objective_catboost(trial, X_train, y_train, X_test, y_test), "CatBoost")

if __name__ == "__main__":
    targets = ['FEDFUNDS', 'GDP', 'CPIAUCSL', 'CUSR0000SAH1', 'CPILFESL', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'HOUST', 
           'DSPI', 'DGS5', 'DGS10', 'AAA', 'BAA', 'WTISPLC', 'IMPGS', 'FGCE', 'PCEPI', 
           'PCEPILFE', 'PAYEMS', 'UNRATE', 'INDPRO', 'CUMFNS', 'USREC', 'DGS2', 'GCE', 'GDPCTPI']

    main(targets)
