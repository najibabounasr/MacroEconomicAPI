import os
import pandas as pd
import numpy as np
import mlflow
import dagshub
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from autogluon.tabular import TabularPredictor
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from funcs.train_model_funcs import clean_data

# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/najibabounasr/MacroEconomicAPI.mlflow")
dagshub.init("MacroEconomicAPI", "najibabounasr", mlflow=True)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'najibabounasr'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'fbaccfb8cf4e8d2d195cd05e9a53dbfe32323695'

# Load data
X_train = pd.read_csv('data/engineered/X_train_engineered.csv', index_col='Date', parse_dates=True)
y_train = pd.read_csv('data/engineered/y_train_engineered.csv', index_col='Date', parse_dates=True)
X_test = pd.read_csv('data/engineered/X_test_engineered.csv', index_col='Date', parse_dates=True)
y_test = pd.read_csv('data/engineered/y_test_engineered.csv', index_col='Date', parse_dates=True)

# Clean data
X_train, y_train = clean_data(X_train, y_train)
X_test, y_test = clean_data(X_test, y_test)

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
        
        mlflow.set_tags({'target_feature': target_feature})

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
            'target_feature': target,
            'feature_addition_rounds': feature_addition_rounds,
            'feature_dropping_threshold': feature_dropping_threshold,
            'fc_parameters': str(fc_parameters)
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
            'fc_parameters': str(fc_parameters)
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
            'fc_parameters': str(fc_parameters)
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
            'fc_parameters': str(fc_parameters)
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

if __name__ == "__main__":
    target_features = ['FEDFUNDS', 'GDP', 'CPIAUCSL', 'CUSR0000SAH1', 'CPILFESL', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 
                       'HOUST', 'DSPI', 'DGS2', 'DGS5', 'DGS10', 'AAA', 'BAA', 'WTISPLC', 'IMPGS', 'GCE', 'FGCE', 
                       'GDPCTPI', 'PCEPI', 'PCEPILFE', 'PAYEMS', 'UNRATE', 'INDPRO', 'CUMFNS', 'USREC']
    
    optimize_all_targets(target_features, X_train, y_train, X_test, y_test)
