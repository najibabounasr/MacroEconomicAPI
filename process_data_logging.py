import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from funcs.process_data_funcs import (
    impute_missing_values_spline, deflate_nominal_values, apply_log_transformations,
    apply_best_transformations, cap_outliers
)

def save_transformed_data(data, transformations_applied, iteration):
    output_dir = 'data/processed/logging'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = f"transformed_data_{iteration}.csv"
    file_path = os.path.join(output_dir, file_name)
    data.to_csv(file_path, index=True)
    print(f"Saved transformed data: {file_path}")

def apply_sliding_window_log(data, window_size=12):
    for col in data.columns:
        for i in range(0, len(data) - window_size + 1):
            window_data = data[col].iloc[i:i + window_size]
            logged_window = np.log1p(window_data)
            data[col].iloc[i:i + window_size] = logged_window
    return data

def apply_selective_logging(data, threshold):
    for col in data.columns:
        data[col] = np.where(data[col] > threshold, np.log1p(data[col]), data[col])
    return data

def apply_relative_transform(data):
    return data.diff().divide(data.shift(1) + 1e-9).dropna()

def apply_local_smoothing(data, window_size=5):
    return data.rolling(window=window_size).mean().dropna()

def apply_soft_clipping(data, threshold, n=1):
    for col in data.columns:
        data[col] = data[col] / (1 + (data[col] / threshold)**n)
    return data

def process_data():
    # Load combined data from raw CSV
    combined_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')
    
    transformation_combinations = [
        ['impute', 'sliding_window_log', 'pct_change'], # EMPTY VAL
        ['impute', 'selective_log', 'pct_change'], # JUST LOOKS THE SAME AS LOCAL SMOOTH, BUT AT A SMALLER SCALE
        ['impute', 'relative_transform', 'pct_change'], # RELATIVE TRANSFORM LOOKS HORRIBLE/ DO NOT USE
        ['impute', 'local_smooth', 'pct_change'], # LOCAL SMOOTH LOOKS FANTASTIC - NOT EXTREMELY SMALL
        ['impute', 'soft_clipping', 'pct_change'], # JUST LOOKS THE SAME AS LOCAL SMOOTH, BUT AT A SMALLER SCALE
        # 9-13: NO PCT_CHANGE
        ['impute', 'sliding_window_log'],
        ['impute', 'selective_log'],
        ['impute', 'relative_transform'],
        ['impute', 'local_smooth'],
        ['impute', 'soft_clipping'],
    ]

    # Iterate through each combination of transformations
    for i, transformations in enumerate(transformation_combinations):
        data = combined_data.copy()
        transformations_applied = []

        # 1. Impute missing values
        if 'impute' in transformations:
            for column in data.columns:
                data = impute_missing_values_spline(data, column)
            transformations_applied.append('impute')

        # 2. Deflate nominal values (optional, depending on your needs)
        if 'deflate' in transformations:
            cpi_col_name = 'CPIAUCSL'
            columns_to_deflate = ['GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 'FGCE', 'DSPI']
            data = deflate_nominal_values(data, cpi_col_name, columns_to_deflate)
            transformations_applied.append('deflate')

        # 3. Sliding Window Log Transformation
        if 'sliding_window_log' in transformations:
            data = apply_sliding_window_log(data)
            transformations_applied.append('sliding_window_log')

        # 4. Selective Log Transformation
        if 'selective_log' in transformations:
            threshold = 1.0  # Choose your threshold here
            data = apply_selective_logging(data, threshold)
            transformations_applied.append('selective_log')

        # 5. Relative Transformations (Normalized Differences)
        if 'relative_transform' in transformations:
            data = apply_relative_transform(data)
            transformations_applied.append('relative_transform')

        # 6. Local Smoothing Before Logging
        if 'local_smooth' in transformations:
            data = apply_local_smoothing(data)
            transformations_applied.append('local_smooth')

        # 7. Soft Clipping for Outliers
        if 'soft_clipping' in transformations:
            threshold = 10  # Set an appropriate threshold
            data = apply_soft_clipping(data, threshold)
            transformations_applied.append('soft_clipping')

        # 8. Apply Percentage Change
        if 'pct_change' in transformations:
            data = data.pct_change().dropna()
            transformations_applied.append('pct_change')

        # Save the transformed data
        save_transformed_data(data, transformations_applied, i)

def main():
    process_data()
    print("Process Data Stage Completed")

if __name__ == "__main__":
    main()
