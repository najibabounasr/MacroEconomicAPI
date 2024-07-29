from funcs.process_data_funcs import fetch_data
import pandas as pd
from local_settings import settings
from fredapi import Fred
import os
from funcs.dvc_funcs import run_dvc_command, dagshub_initialization, upload_to_dagshub
from funcs.api_funcs import get_target_arg
from dagshub import get_repo_bucket_client
import sys

def fetch_all_data(target_feature):
    # Initialize FRED API with your API key
    fred = Fred(api_key=settings['api_key'])

    # Fetch and store data for each series in a dictionary
    data_frames = {}
    for series_id in settings['series_ids']:
        frequency = settings['frequency_map'].get(series_id, 'm')  # Default to 'm' if not specified
        try:
            data_frame = fetch_data(series_id, frequency)
            if not data_frame.empty:
                data_frames[series_id] = data_frame
        except Exception as e:
            print(f"Error fetching data for {series_id}: {e}")

    # Combine all data into a single DataFrame
    combined_data = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())
    combined_data = combined_data.asfreq('MS')
    combined_data.index.name = 'Date'

    # Drop a level from the multi-level columns
    combined_data.columns = combined_data.columns.droplevel(1)

    # Ensure the target feature is included in the dataset
    if target_feature not in combined_data.columns:
        valid_features = ", ".join(combined_data.columns)
        raise ValueError(f"Target feature '{target_feature}' is not available in the dataset. Valid features are: {valid_features}")

    # Save raw data locally
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    combined_data.to_csv('data/raw/raw_data.csv')

    # Upload raw data to Dagshub
    s3 = get_repo_bucket_client("najibabounasr/MacroEconomicAPI")
    s3.upload_file(
        Bucket="MacroEconomicAPI",
        Filename="data/raw/raw_data.csv",
        Key="data/raw/raw_data.csv",
    )

    print("Data fetching complete and tracked with DVC and Dagshub.")
    return combined_data

def main():
    dagshub_initialization()
    if len(sys.argv) < 2:
        raise ValueError("No target feature provided. Please specify the target feature.")
    target_feature = get_target_arg()
    fetch_all_data(target_feature)
    print("Fetch Data Stage Completed")

if __name__ == "__main__":
    main()
