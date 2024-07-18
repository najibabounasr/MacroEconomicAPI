import os
import pandas as pd
import numpy as np
import configparser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from funcs.process_data_funcs import (
    impute_missing_values_spline, deflate_nominal_values, apply_log_transformations,
    apply_best_transformations, cap_outliers
)
from funcs.dvc_funcs import run_dvc_command, dagshub_initialization, load_dvc_config, check_remote_config, verify_dvc_remote
from dagshub import get_repo_bucket_client
import sys
def main(target):
    # Load combined data from DVC
    combined_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')

    # Perform train/test split
    X = combined_data.drop(columns=[target])
    y = combined_data[[target]]

    # Ensure y retains 'Date' index
    y.index.name = 'Date'

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Convert index to datetime if not already
    X_train.index = pd.to_datetime(X_train.index)
    X_test.index = pd.to_datetime(X_test.index)
    y_train.index = pd.to_datetime(y_train.index)
    y_test.index = pd.to_datetime(y_test.index)

    # Impute missing values
    quarterly_columns = ['GDP', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'GDPCTPI']
    treasury_yield_columns = ['DGS2', 'DGS5', 'DGS10']

    for column in quarterly_columns + treasury_yield_columns:
        if column in X_train.columns:
            X_train = impute_missing_values_spline(X_train, column)
            X_test = impute_missing_values_spline(X_test, column)
        elif column in y_train.columns:  # Handle case where target is one of the columns
            y_train = impute_missing_values_spline(y_train, column)
            y_test = impute_missing_values_spline(y_test, column)
        else:
            print(f"Column {column} does not exist in DataFrame.")

    # Verify imputed values
    for column in quarterly_columns + treasury_yield_columns:
        if column in X_train.columns:
            print(f"Imputed {column} - Correlation with original X_train: {X_train[column].corr(X_train[column])}")
            print(f"Imputed {column} - Correlation with original X_test: {X_test[column].corr(X_test[column])}")
        elif column in y_train.columns:  # Handle case where target is one of the columns
            print(f"Imputed {column} - Correlation with original y_train: {y_train[column].corr(y_train[column])}")
            print(f"Imputed {column} - Correlation with original y_test: {y_test[column].corr(y_test[column])}")

    # Name the index column Date
    for df in [X_train, X_test, y_train, y_test]:
        df.index.name = 'Date'

    # Combine X and y dataframes
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    # Ensure CPIAUCSL is included in the dataframes
    cpi_col_name = 'CPIAUCSL'
    columns_to_deflate = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'DSPI']

    # Handle deflation based on conditions
    if target == cpi_col_name:
        # Deflate CPIAUCSL if it is the target
        train_combined[target] = deflate_nominal_values(train_combined[[target]], cpi_col_name, [target])
        test_combined[target] = deflate_nominal_values(test_combined[[target]], cpi_col_name, [target])
    elif target in columns_to_deflate:
        # Deflate the target column if it is in the columns to deflate list
        train_combined[target] = deflate_nominal_values(train_combined[[target, cpi_col_name]], cpi_col_name, [target])
        test_combined[target] = deflate_nominal_values(test_combined[[target, cpi_col_name]], cpi_col_name, [target])
        # Remove the target from the list to avoid double deflation
        columns_to_deflate.remove(target)
        # Deflate the remaining columns
        train_combined = deflate_nominal_values(train_combined, cpi_col_name, columns_to_deflate)
        test_combined = deflate_nominal_values(test_combined, cpi_col_name, columns_to_deflate)
    else:
        # Deflate the remaining columns
        train_combined = deflate_nominal_values(train_combined, cpi_col_name, columns_to_deflate)
        test_combined = deflate_nominal_values(test_combined, cpi_col_name, columns_to_deflate)

    # Apply logarithmic transformations
    columns_to_transform = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'HOUST', 'DSPI']

    # Apply log transformation to target if it is in the columns to transform
    if target in columns_to_transform:
        train_combined[target] = apply_log_transformations(train_combined[[target]], [target])
        test_combined[target] = apply_log_transformations(test_combined[[target]], [target])
        columns_to_transform.remove(target)

    train_combined = apply_log_transformations(train_combined, columns_to_transform)
    test_combined = apply_log_transformations(test_combined, columns_to_transform)

    # Separate X and y after transformations
    X_train = train_combined.drop(columns=[target])
    y_train = train_combined[[target]]
    X_test = test_combined.drop(columns=[target])
    y_test = test_combined[[target]]

    # Standardize/Normalize the Data
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)
    y_train[target] = scaler.fit_transform(y_train[target].values.reshape(-1, 1))
    y_test[target] = scaler.transform(y_test[target].values.reshape(-1, 1))

    # Apply Percentage Change
    X_train_pct_change = X_train.pct_change().dropna()
    X_test_pct_change = X_test.pct_change().dropna()
    y_train_pct_change = y_train.pct_change().dropna()
    y_test_pct_change = y_test.pct_change().dropna()

    # Apply the best transformations
    X_train_transformed = apply_best_transformations(X_train_pct_change)
    X_test_transformed = apply_best_transformations(X_test_pct_change)
    y_train_transformed = apply_best_transformations(y_train_pct_change)
    y_test_transformed = apply_best_transformations(y_test_pct_change)

    # Ensure y_train_transformed and y_test_transformed have the correct column name
    y_train_transformed.columns = [target]
    y_test_transformed.columns = [target]

    # Handle outliers in the transformed data
    X_train_transformed = cap_outliers(X_train_transformed, cap_factor=3.0)
    y_train_transformed = cap_outliers(y_train_transformed, cap_factor=3.0)

    # Combine X and y after final transformations
    train_transformed_combined = pd.concat([X_train_transformed, y_train_transformed], axis=1)
    test_transformed_combined = pd.concat([X_test_transformed, y_test_transformed], axis=1)

    # Save the transformed data
    train_transformed_combined.to_csv('data/processed/train_transformed_combined.csv', index=True)
    test_transformed_combined.to_csv('data/processed/test_transformed_combined.csv', index=True)

    # Upload to Dagshub storage
    s3 = get_repo_bucket_client("najibabounasr/MacroEconomicAPI")
    s3.upload_file(
        Bucket="MacroEconomicAPI",  # name of the repo
        Filename="data/processed/train_transformed_combined.csv",  # local path of file to upload
        Key="train_transformed_combined.csv",  # remote path where to upload the file
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",  # name of the repo
        Filename="data/processed/test_transformed_combined.csv",  # local path of file to upload
        Key="test_transformed_combined.csv",  # remote path where to upload the file
    )

    # Save individual transformed datasets
    X_train_transformed.to_csv('data/processed/X_train_transformed.csv', index=True)
    X_test_transformed.to_csv('data/processed/X_test_transformed.csv', index=True)
    y_train_transformed.to_csv('data/processed/y_train_transformed.csv', index=True)
    y_test_transformed.to_csv('data/processed/y_test_transformed.csv', index=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise ValueError("No target feature provided. Please specify the target feature.")
    target = sys.argv[1]
    dagshub_initialization()
    main(target)
