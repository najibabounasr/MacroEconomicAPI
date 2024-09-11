import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from funcs.process_data_funcs import (
    impute_missing_values_spline, deflate_nominal_values, apply_log_transformations,
    apply_best_transformations, cap_outliers
)
from funcs.dvc_funcs import dagshub_initialization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
# import quantile and power transformer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
def save_transformed_data(data, transformations_applied, iteration):
    # Create the output directory if it doesn't exist
    output_dir = 'data/processed/testing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the file name based on the transformations applied
    file_name = f"transformed_data_{iteration}_" + ".csv"
    file_path = os.path.join(output_dir, file_name)

    # Save the transformed data
    data.to_csv(file_path, index=True)
    print(f"Saved transformed data: {file_path}")


def process_data():
    # Load combined data from raw CSV
    combined_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')
    
    # Separate 'USREC' before processing
    # usrec_data = combined_data['USREC']
    # combined_data = combined_data.drop(columns=['USREC'])  # Exclude categorical column

    # Define all transformation combinations to try
    transformation_combinations = [
        ['impute', 'deflate', 'log', 'scale', 'pct_change', 'adf', 'cap'], # CAN USE
        ['impute', 'deflate', 'log', 'scale', 'pct_change', 'adf'], # CAN USE
        ['impute', 'deflate', 'log', 'scale', 'pct_change'], # / CAN USE
        ['impute', 'deflate', 'log', 'scale','pct_change'], # / CAN USE
        ['impute', 'deflate', 'log','pct_change'], # DON'T USE/ WHY LOG SOME IF WE CAN LOG ALL # NO USE / CAN USE
        ['impute', 'deflate','pct_change'], # 5. # WORKS, BUT MIGHT NOT BE THE BEST CHOICE # INCLUDE STILL / CAN USE
        ['impute', 'pct_change'], # WORKS - BUT SIMPLE / CAN USE
        ['impute','scale','pct_change','adf'],# MID / CAN USE
        ['impute','scale','adf','pct_change'], # MID / CAN USE
        ['impute','adf','scale','pct_change'], # MID / CAN USE
        ['impute','pct_change','adf'], # # 10. # MID / CAN USE
        ['impute','adf','pct_change'], # MID / CAN USE
        ['impute','adf','pct_change','scale'], # SCALING AFTER PCT_CHANGE DOESNT WORK FOR NOW # NO USE
        ['impute','adf','scale'], # NO WORK # NO USE
        ['impute','pct_change','scale'], # NO WORK # NO USE
        ['impute', 'adf', 'scale', 'pct_change'], # 15 # INTRRODUCES NAN # NO USE
        ['impute','adf','pct_change','robust'], # NO WORK # NO USE
        ['impute','pct_change','robust'],# NO WORK # NO USE
        ['impute','pct_change','power'], # NO WORK # NO USE
        ['impute','pct_change','quantile'], # WORKS, SKEWS DATA # NO USE
        ['impute','log_all','pct_change','scale'], # 20 # APPLYING ANY OF THE THREE TRANSFORMATION METHODS, EVEN WHEN LOGGED BEFORE, STILL LEADS TO OUTLIERS AND ANOMALIES. 
        ['impute','log_all','pct_change','robust'], # 21
        ['impute','log_all','pct_change','power'], # NANS
        ['impute','log_all','adf','pct_change'], # NANS
        ['impute','log_all'], # 24 TRY THIS / CAN USE
        ['impute','log_all','pct_change'], # 25 TRY TIS / CAN USE
        ['impute'], # 26 TRY THIS / CAN USE
        ['impute','log_all','impute','pct_change'], # 27 / CAN USE
        ['impute','log_all','impute','pct_change','impute','scale'], # 28/ CAN USE
        ['impute','log_all','impute','pct_change','impute','scale','impute','impute'], # 29 / CAN USE

        # Add more combinations based on your requirements
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

        # 2. Deflate nominal values
        if 'deflate' in transformations:
            cpi_col_name = 'CPIAUCSL'
            columns_to_deflate = [
                'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 'GCE', 
                'FGCE', 'DSPI'
            ]
            data = deflate_nominal_values(data, cpi_col_name, columns_to_deflate)
            transformations_applied.append('deflate')

        # 3. Apply logarithmic transformations
        if 'log' in transformations:
            columns_to_transform = [
                'GDP', 'PCE', 'PRFI', 'PNFI', 'EXPGS', 'IMPGS', 
                'GCE', 'FGCE', 'HOUST', 'DSPI', 'M1', 'M1V', 'M2', 
                'WTISPLC'
            ]
            data = apply_log_transformations(data, columns_to_transform)
            transformations_applied.append('log')

        # 4. Standardize/Normalize the Data
        if 'scale' in transformations:
            scaler = StandardScaler()
            data[data.columns] = scaler.fit_transform(data)
            transformations_applied.append('scale')

        # 5. Apply Percentage Change
        if 'pct_change' in transformations:
            data = data.pct_change().dropna()
            transformations_applied.append('pct_change')

        # 6. Apply ADF-based transformations
        if 'adf' in transformations:
            data = apply_best_transformations(data)
            transformations_applied.append('adf')

        # 7. Cap outliers
        if 'cap' in transformations:
            data = cap_outliers(data, cap_factor=3.0)
            transformations_applied.append('cap')

        # 8. Robust Scaler
        if 'robust' in transformations:
            scaler = RobustScaler()
            data[data.columns] = scaler.fit_transform(data)
            transformations_applied.append('robust')

        # 9. Quantile Transformer
        if 'quantile' in transformations:
            scaler = QuantileTransformer()
            data[data.columns] = scaler.fit_transform(data)
            transformations_applied.append('quantile')
        
        # 10. Power Transformer
        if 'power' in transformations:
            scaler = PowerTransformer()
            data[data.columns] = scaler.fit_transform(data)
            transformations_applied.append('power')
        
        # 11. Log All
        if 'log_all' in transformations:
            columns_to_transform = data.columns
            data = apply_log_transformations(data, columns_to_transform)
            transformations_applied.append('log_all')
        # # Add 'USREC' back to the dataset after processing
        # data['USREC'] = usrec_data.reindex(data.index)

        # Save the transformed data
        save_transformed_data(data, transformations_applied, i)


def main():
    process_data()
    print("Process Data Stage Completed")


if __name__ == "__main__":
    main()
