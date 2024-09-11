# Reversing the transformations
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from funcs.api_funcs import get_target_arg, get_feature_addition_rounds_arg, get_feature_dropping_threshold_arg, get_tsfresh_fc_params_arg
from sklearn.neighbors import KNeighborsRegressor
from autogluon.tabular import TabularPredictor
from funcs.train_model_funcs import clean_data
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

targets = ['M1REAL','WTISPLC','CPIAUCSL','HOUST']
features = data.columns
files = os.listdir('data/processed/testing')
xgboost_params = {'max_depth': 9, 'learning_rate': 0.016810144010995204, 'n_estimators': 193, 'min_child_weight': 3, 'subsample': 0.5865511380196138, 'colsample_bytree': 0.9177433017782509, 'reg_alpha': 0.012513778476694921, 'reg_lambda': 8.577576991968611e-05}
lightgbm_params = {'num_leaves': 119, 'learning_rate': 0.0832469304666387, 'n_estimators': 71, 'max_depth': 3, 'min_child_samples': 18, 'subsample': 0.7402007213047204, 'colsample_bytree': 0.9405245932820381, 'reg_alpha': 0.000756445935013929, 'reg_lambda': 0.0002590470172405959}

bst = XGBRegressor(**xgboost_params)
lgbm = LGBMRegressor(**lightgbm_params)

xgb_model_type_s = pd.Series(dtype=str)
xgb_mse_score_s = pd.Series(dtype=float)
xgb_rmse_score_s = pd.Series(dtype=float)
xgb_CV_round_s = pd.Series(dtype=int)
xgb_target_s = pd.Series(dtype=str)
xgbm_file_s  = pd.Series(dtype=str)
lgbm_model_type_s = pd.Series(dtype=str)
lgbm_mse_score_s = pd.Series(dtype=float)
lgbm_rmse_score_s = pd.Series(dtype=float)
lgbm_CV_round_s = pd.Series(dtype=int)
lgbm_target_s = pd.Series(dtype=str)
lgbm_file_s  = pd.Series(dtype=str)

for file in files:
    # Step 1: Use predictive models to create predictions on the data we used
    data= pd.read_csv('data/processed/testing/'+file,index_col='Date',parse_dates=True)

    # Step 2: creating a list of transformations, based on file names:
    transformations = []
    transformation_strings = ['impute','deflate','log','scale','pct_change','adf','cap','robust','quantile','power','log_all','sliding_window_log','selective_log','soft_clipping','local_smooth','relative_transform']
    for string in transformation_strings:
        if string in file:
            transformations.append(string)
    print(f"transformations list: {transformations}")
    columns_w_inf = []
    print(f"file : {file}")
    for col in data.columns:
        mean = data[col].mean()
        for i in range(len(data)):
            if data.iloc[i][col] > mean*1000000000000:
                if col not in columns_w_inf:
                    columns_w_inf.append(col)
                    # print("Column",col,"has a value too large, on row",i)
    features = list(data.columns)
    new_list = features
    for item in columns_w_inf:
        # print(f"Column: {item} was found with overly large or inf values")
        new_list.remove(item)
    data = data[new_list]
    for item in columns_w_inf:
        if item in new_list:
            new_list.remove(item)
    data = data[new_list]
    for target in targets:
        if target not in new_list:
            # print(f"NEW LIST:",new_list)
            # print(f"target: {target}")
            # print("SKIPPED NOW")
            continue
        for tr_index, val_index in tscv.split(data):
            X_train = data.loc[:,data.columns!= target].iloc[tr_index]
            # display(tr_index)
            # display(val_index)
            X_test = data.loc[:,data.columns!= target].iloc[val_index]
            y_test = data[target].iloc[val_index]
            y_train = data[target].iloc[tr_index]
            bst.fit(X_train,y_train)
            xgb_preds = bst.predict(X_test)
            # print(xgb_preds)
            xgb_mse = mean_squared_error(xgb_preds,y_test)
            xgb_rmse = (xgb_mse)**0.5
            #### PUT REVERSAL OF DATA LOGIC, HERE




    



    # Step 2.5 prepare for reversals
    raw_data = pd.read_csv('data/raw/raw_data.csv',index_col='Date',parse_dates=True)
    file_name = f"transformed_data__"+"__".join(transformations)+".csv"
    file_path = os.path.join(output_dir, file_name)
    transformed_data = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    print(f"transformed data: {transformed_data.head()}")







    # Step 3: perform multiple reversals
        # 1. Impute missing values
    if 'impute' in transformations:
        skip = 'yes'
        # Since we cannot have NaN's, we will never reverse the imputation of data

    # 2. Deflate nominal values
    if 'deflate' in transformations:
        cpi_col_name = 'CPIAUCSL'
        columns_to_deflate = [
            'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 
            'FGCE', 'DSPI'
        ]
        # We now reverse the deflation
        data = reverse_deflate_nominal_values(data, cpi_col_name, columns_to_deflate)
        transformations_applied.append('deflate')

    # 3. Apply logarithmic transformations
    if 'log' in transformations:
        columns_to_transform = [
            'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 
            'GCE', 'FGCE', 'HOUST', 'DSPI', 'M1', 'M1V', 'M2', 
            'WTISPLC'
        ]
        data = reverse_log_transformations(data, columns_to_transform)
        transformations_applied.append('log')

    # 4. Standardize/Normalize the Data
    if 'scale' in transformations:
        raw_data = pd.read_csv(raw_data_path, parse_dates=True, index_col='Date')
        
        # Initialize a scaler (even though we don't use it for transforming)
        scaler = StandardScaler()
        
        # Fit the scaler on the original raw data to get the original mean and std
        scaler.fit(raw_data[data.columns])
        
        # Reverse the scaling using the mean and scale (std) from the original raw data
        data[data.columns] = scaler.inverse_transform(data)


    # 5. Apply Percentage Change
    if 'pct_change' in transformations:
        for col in data.columns:
            initial_value = data[col].iloc[0]

            reversed_series = (data[col] + 1).cumprod() * initial_value
            reversed_data[col] = reversed_series
            reversed_data.ffill()
        transformations_applied.append('pct_change')

    # 6. Apply ADF-based transformations
    if 'adf' in transformations:
        data = reverse_best_transformations(data)
        transformations_applied.append('adf')

    # 7. Cap outliers
    if 'cap' in transformations:
        skip = 'yes'
        # We can't really do anything
        # data = cap_outliers(data, cap_factor=3.0)
        # transformations_applied.append('cap')

    # 8. Robust Scaler
    if 'robust' in transformations:
        scaler = RobustScaler()
        scaler.fit(raw_data[data.columns])  # Fit on the original data
        data[data.columns] = scaler.inverse_transform(data)


    # 9. Quantile Transformer
    if 'quantile' in transformations_applied:
        scaler = QuantileTransformer()
        scaler.fit(raw_data[data.columns])  # Fit on the original data
        data[data.columns] = scaler.inverse_transform(data)

    
    # 10. Power Transformer
    if 'power' in transformations_applied:
        scaler = PowerTransformer()
        scaler.fit(raw_data[data.columns])  # Fit on the original data
        data[data.columns] = scaler.inverse_transform(data)

    
    # 11. Log All
    if 'log_all' in transformations:
        columns_to_transform = data.columns
        data = reverse_log_transformations(data, columns_to_transform)
        transformations_applied.append('log_all')
    
    # 12. Sliding Window Log Transformation
    if 'sliding_window_log' in transformations:
        data = reverse_sliding_window_log(data)
        transformations_applied.append('sliding_window_log')

    # 13. Selective Log Transformation
    if 'selective_log' in transformations:
        threshold = 1.0  # Choose your threshold here
        data = reverse_selective_logging(data, threshold)
        transformations_applied.append('selective_log')

    # 14. Relative Transformations (Normalized Differences)
    if 'relative_transform' in transformations:
        data = reverse_relative_transform(data)
        transformations_applied.append('relative_transform')

    # 15. Local Smoothing Before Logging
    if 'local_smooth' in transformations:
        skip = 'yes'
        # we cannot really reverse local smooth, just leave it alone, skip it, return nothing
        # data = reverse_local_smoothing(data)
        # transformations_applied.append('local_smooth')

    # 16. Soft Clipping for Outliers
    if 'soft_clipping' in transformations:
        threshold = 10  # Set an appropriate threshold
        data = reverse_soft_clipping(data, threshold)
        transformations_applied.append('soft_clipping')
# # Test the data after reversing the transformations
# for transformation in transformation_combinations:
#     # 2. Split the data into features and target
#     targets = ['M1','WTISPLC','CPIAUCSL','HOUST']
