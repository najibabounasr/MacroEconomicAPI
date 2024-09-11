import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from funcs.process_data_funcs import (
    impute_missing_values_spline, deflate_nominal_values, apply_log_transformations,
    apply_best_transformations, cap_outliers
)
from funcs.dvc_funcs import dagshub_initialization
from dagshub import get_repo_bucket_client
from sklearn.model_selection import train_test_split
def process_data():
    # Initialize Dagshub and DVC
    dagshub_initialization()

    # Load combined data from DVC
    combined_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')
    # combined_data = combined_data.drop(columns=['USREC'])
    
    # 1. Impute missing values
    # Impute missing values for all columns
    for column in combined_data.columns:
        combined_data = impute_missing_values_spline(combined_data, column)

    # Deflate nominal values
    # 2. Deflate nominal values
    # cpi_col_name = 'CPIAUCSL'
    columns_to_deflate = [
        'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 
        'FGCE', 'DSPI'
    ]
    combined_data = deflate_nominal_values(combined_data, cpi_col_name, columns_to_deflate)

    # Apply logarithmic transformations
    columns_to_transform = [
        'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 
        'GCE', 'FGCE', 'HOUST', 'DSPI'
    ]
    combined_data = apply_log_transformations(combined_data, columns_to_transform)

    # Standardize/Normalize the Data
    scaler = StandardScaler()
    combined_data[combined_data.columns] = scaler.fit_transform(combined_data)

    # Apply Percentage Change (for the entire dataset)
    combined_data = combined_data.pct_change().dropna()

    # Apply the best transformations (ADF-based)
    combined_data = apply_best_transformations(combined_data)

    # # Cap outliers in the transformed data
    combined_data = cap_outliers(combined_data, cap_factor=3.0)
    target = 'FEDFUNDS'

    # Perform train/test split using a random column
    train_combined, test_combined = train_test_split(combined_data, test_size=0.2, random_state=42, shuffle=False)
    # Split into Train/Test datasets (at the end)
    train_data = combined_data[:int(0.8 * len(combined_data))]
    test_data = combined_data[int(0.8 * len(combined_data)):]

    # Save the transformed data locally
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    # train_data.to_csv('data/processed/train_transformed_combined.csv', index=True)
    # test_data.to_csv('data/processed/test_transformed_combined.csv', index=True)

    # Ensure the index column 'Date' is properly named
    for df in [train_combined, test_combined]:
        df.index.name = 'Date'
    # Upload to Dagshub storage
    s3 = get_repo_bucket_client("najibabounasr/MacroEconomicAPI")
    s3.upload_file(
        Bucket="MacroEconomicAPI",  # name of the repo
        Filename="data/processed/train_transformed_combined.csv",  # local path of file to upload
        Key="data/processed/train_transformed_combined.csv",  # remote path where to upload the file
    )
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/processed/test_transformed_combined.csv",
        Key="data/processed/test_transformed_combined.csv",
    )

def main():
    process_data()
    print("Process Data Stage Completed")

if __name__ == "__main__":
    main()
