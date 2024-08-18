import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from funcs.process_data_funcs import (
    impute_missing_values_spline, deflate_nominal_values, apply_log_transformations,
    apply_best_transformations, cap_outliers
)
from funcs.dvc_funcs import dagshub_initialization
from dagshub import get_repo_bucket_client

def process_data_for_target(target, combined_data):
    # Perform train/test split
    X = combined_data.drop(columns=[target])
    y = combined_data[[target]]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Impute missing values
    columns_to_impute = X_train.columns  # Impute all columns
    for column in columns_to_impute:
        X_train = impute_missing_values_spline(X_train, column)
        X_test = impute_missing_values_spline(X_test, column)
    y_train = impute_missing_values_spline(y_train, target)
    y_test = impute_missing_values_spline(y_test, target)

    # Name the index column 'Date'
    for df in [X_train, X_test, y_train, y_test]:
        df.index.name = 'Date'

    # Combine X and y dataframes
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)

    # Deflate nominal values
    cpi_col_name = 'CPIAUCSL'
    columns_to_deflate = list(set(X_train.columns).intersection(
        ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'DSPI']
    ))
    if target in columns_to_deflate:
        deflated_train = deflate_nominal_values(train_combined[[target, cpi_col_name]], cpi_col_name, [target])
        deflated_test = deflate_nominal_values(test_combined[[target, cpi_col_name]], cpi_col_name, [target])
        train_combined[target] = deflated_train[target]
        test_combined[target] = deflated_test[target]
        columns_to_deflate.remove(target)
    
    train_combined = deflate_nominal_values(train_combined, cpi_col_name, columns_to_deflate)
    test_combined = deflate_nominal_values(test_combined, cpi_col_name, columns_to_deflate)

    # Apply logarithmic transformations
    columns_to_transform = list(set(X_train.columns).intersection(
        ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'HOUST', 'DSPI']
    ))
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

    # Apply Percentage Change (after splitting to avoid data leakage)
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

    # Save the transformed data locally in a subdirectory for the target
    save_path = f'data/processed/{target}_transformed'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_transformed_combined.to_csv(f'{save_path}/train_transformed_combined.csv', index=True)
    test_transformed_combined.to_csv(f'{save_path}/test_transformed_combined.csv', index=True)

    # Upload to Dagshub storage
    s3 = get_repo_bucket_client("najibabounasr/MacroEconomicAPI")
    s3.upload_file(
        Bucket="MacroEconomicAPI",  # name of the repo
        Filename=f'{save_path}/train_transformed_combined.csv',  # local path of file to upload
        Key=f'data/processed/{target}/train_transformed_combined.csv',  # remote path where to upload the file
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename=f'{save_path}/test_transformed_combined.csv',
        Key=f'data/processed/{target}/test_transformed_combined.csv',
    )

def main():
    # Initialize Dagshub and DVC
    dagshub_initialization()

    # Load combined data from DVC
    combined_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')

    # Process data for each feature as a target
    for target in combined_data.columns:
        process_data_for_target(target, combined_data)
        print(f"Processed data for target: {target}")

if __name__ == "__main__":
    main()
    print("Process Data Stage Completed")
