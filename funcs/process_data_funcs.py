# Import the log_transformed_df.csv file from the data folder
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.stattools import adfuller
# from funcs.machine_learning import check_stationarity, plot_series_stationarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from local_settings import settings
from fredapi import Fred
import requests

# Use the settings dictionary
api_key = settings['api_key']
series_ids = settings['series_ids']
start_date = settings['start_date']
end_date = settings['end_date']
# Base URL for API requests
base_url = 'https://api.stlouisfed.org/fred/series/observations'
# Initialize the FRED API with your API key
fred = Fred(api_key=settings['api_key'])
# from funcs.loading_csv_functions import merge_new_data, merge_new_data_and_apply_pct_change, prepare_cpi_data, preprocess_and_merge
# from funcs.loading_csv_functions import load_and_process_cpi_data
def name(self) -> any:
    return self.attribute
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hvplot.pandas  # Import HvPlot for Pandas
import matplotlib.pyplot as plt # Import show function from Bokeh
from statsmodels.tsa.arima.model import ARIMA
import pickle

#############FUNCTIONS: ###############################################################################3
# Function to fetch and prepare data
def fetch_data(series_id,frequency):
    try:
        print(f"Fetching data for {series_id}")
        data = fred.get_series(series_id, observation_start=settings['start_date'], observation_end=settings['end_date'],frequency=frequency)
        data.index = pd.to_datetime(data.index)  # Convert index to datetime
        return pd.DataFrame(data, columns=[series_id])
    except Exception as e:
        print(f"Error fetching data for {series_id}: {str(e)}")
        return pd.DataFrame()
def deflate_nominal_values(df, cpi_col_name, columns_to_deflate):
    """
    Deflates the nominal values in the specified columns of the dataframe using the CPI column.

    :param df: DataFrame containing the columns to deflate and the CPI column
    :param cpi_col_name: Name of the CPI column
    :param columns_to_deflate: List of column names to deflate
    :return: DataFrame with deflated values in the specified columns
    """
    for col in columns_to_deflate:
        df.loc[:, col] = df[col] / df[cpi_col_name] * 100
    return df
def apply_log_transformations(df, columns_to_transform):
    for col in columns_to_transform:
        # Protect against non-positive values by applying log1p to only positive values
        df[col] = np.where(df[col] > 0, 100 * np.log1p(df[col]), df[col])
        # Handle NaNs by forward-filling and backward-filling
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)
    return df

def cap_outliers(df, cap_factor=3.0):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - cap_factor * IQR
        upper_bound = Q3 + cap_factor * IQR
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df
def check_stationarity(data):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity.
    
    Arguments:
    Pandas Series: a series of data to be checked for stationarity.
    
    Returns:
    Prints test statistics and critical values.
    """
    # Perform Augmented Dickey-Fuller test
    # Perform the test using the AIC criterion for choosing the number of lags
    print('Results of Augmented Dickey-Fuller Test:')
    adf_test = adfuller(data, autolag='AIC')  

    # Extract and print the test statistics and critical values
    adf_output = pd.Series(adf_test[0:4], 
                           index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)
    return adf_output


import matplotlib.pyplot as plt

def plot_series_stationarity(series, window=12):
    """
    Plot the time series, its rolling mean, and its rolling standard deviation.
    
    Arguments:
    series: Pandas Series - the time series to plot.
    window: int - the window size for calculating rolling statistics.
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plot the statistics
    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Original Series')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std Dev')
    plt.title('Time Series Stationarity Check')
    plt.legend()
    plt.show()
import itertools
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hvplot.pandas  # Import HvPlot for Pandas
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import dim, opts
from bokeh.plotting import show  # Import show function from Bokeh
from statsmodels.tsa.arima.model import ARIMA
import pickle

def clean_data(df):
    """
    Cleans the input DataFrame by:
    - Replacing infinities with NaN
    - Filling NaN values using backfill and forward fill
    - Interpolating any remaining NaN values
    - Ensuring all columns are numeric
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.interpolate(method='linear', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().values.any():
        df.fillna(df.mean(), inplace=True)
    return df
def check_stationarity(data):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity.
    
    Arguments:
    Pandas Series: a series of data to be checked for stationarity.
    
    Returns:
    Prints test statistics and critical values.
    """
    # Perform Augmented Dickey-Fuller test
    # Perform the test using the AIC criterion for choosing the number of lags
    print('Results of Augmented Dickey-Fuller Test:')
    adf_test = adfuller(data, autolag='AIC')  

    # Extract and print the test statistics and critical values
    adf_output = pd.Series(adf_test[0:4], 
                           index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    print(adf_output)
    return adf_output
def plot_series_stationarity(series, window=12):
    """
    Plot the time series, its rolling mean, and its rolling standard deviation.
    
    Arguments:
    series: Pandas Series - the time series to plot.
    window: int - the window size for calculating rolling statistics.
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plot the statistics
    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Original Series')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std Dev')
    plt.title('Time Series Stationarity Check')
    plt.legend()
    plt.show()

from scipy.interpolate import CubicSpline
def impute_missing_values_spline(df, column):
    # Ensure the index is in datetime format and sort the data
    df = df.sort_index()
    # Extract the non-missing values to fit the spline
    known_data = df.dropna(subset=[column])
    known_index = known_data.index.map(pd.Timestamp.toordinal)  # Convert dates to ordinal
    # Fit a cubic spline using known data points
    cs = CubicSpline(known_index, known_data[column])
    # Apply the cubic spline to predict missing values
    missing_index = df[df[column].isnull()].index.map(pd.Timestamp.toordinal)
    predicted_values = cs(missing_index)
    # Fill in the missing values in the original DataFrame
    df.loc[df[column].isnull(), column] = predicted_values
    return df


def evaluate_transformations(series):
    methods = {
        'None': series,
        'Simple Differencing': series.diff().dropna(),
        'Rolling Mean Subtraction': (series - series.rolling(window=7).mean()).dropna(),
        'Rolling Mean Subtraction + Differencing': (series - series.rolling(window=7).mean()).diff().dropna()
    }           

    results = {}
    for method, transformed_series in methods.items():
        adf_result = adfuller(transformed_series)
        results[method] = (adf_result[0], adf_result[1])  # Storing the ADF statistic and p-value

    best_method = min(results, key=lambda x: results[x][0])  # Find the method with the smallest ADF statistic
    return best_method, results[best_method]

def apply_best_transformations(df):
    transformed_df = pd.DataFrame(index=df.index)
    transformation_results = {}
    for column in df.columns:
        series_data = df[column].dropna()  # Ensure no NaN values which might cause issues in computations
        best_method, (best_statistic, _) = evaluate_transformations(series_data)
        transformation_results[column] = {'Best Method': best_method, 'ADF Statistic': best_statistic}

        # Print statement to declare the column and the best transformation
        # print(f"Column: {column}, Best Method: {best_method}, ADF Statistic: {best_statistic}")
        
        if best_method == 'Simple Differencing':
            transformed_df[column] = df[column].diff().bfill()
        elif best_method == 'Rolling Mean Subtraction':
            rolling_mean = df[column].rolling(window=7).mean()
            transformed_df[column] = (df[column] - rolling_mean).bfill()
        elif best_method == 'Rolling Mean Subtraction + Differencing':
            rolling_mean = df[column].rolling(window=7).mean()
            transformed_df[column] = (df[column] - rolling_mean).diff().bfill()
        else:
            transformed_df[column] = df[column]
    
    transformation_results_df = pd.DataFrame(transformation_results).T
    transformation_results_df.to_csv('best_transformations.csv')
    return transformed_df

def apply_sliding_window_log(data, window_size=12):
    for col in data.columns:
        # Apply sliding window logging while avoiding inplace operations that may cause unintended issues
        for i in range(window_size - 1, len(data)):
            window_data = data[col].iloc[i - window_size + 1: i + 1]
            logged_window = np.log1p(window_data)
            # Update only this window
            data[col].iloc[i - window_size + 1: i + 1] = logged_window
        # Forward and backward fill for any NaNs introduced
        data[col].fillna(method='ffill', inplace=True)
        data[col].fillna(method='bfill', inplace=True)
    return data

def apply_selective_logging(data, threshold):
    # Apply log1p only to values above the threshold
    data = np.where(data > threshold, np.log1p(data), data)
    # Fill NaNs to handle any missing data
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data


def apply_relative_transform(data):
    return data.diff().divide(data.shift(1) + 1e-9).dropna()

def apply_local_smoothing(data, window_size=5):
    return data.rolling(window=window_size).mean().dropna()

def apply_soft_clipping(data, threshold, n=1):
    for col in data.columns:
        data[col] = data[col] / (1 + (data[col] / threshold)**n)
    return data


import itertools
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hvplot.pandas  # Import HvPlot for Pandas
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import dim, opts
from bokeh.plotting import show  # Import show function from Bokeh
from statsmodels.tsa.arima.model import ARIMA
import pickle

def deflate_nominal_values(df, cpi_col_name, columns_to_deflate):
    """
    Deflates the nominal values in the specified columns of the dataframe using the CPI column.

    :param df: DataFrame containing the columns to deflate and the CPI column
    :param cpi_col_name: Name of the CPI column
    :param columns_to_deflate: List of column names to deflate
    :return: DataFrame with deflated values in the specified columns
    """
    for col in columns_to_deflate:
        df.loc[:, col] = df[col] / df[cpi_col_name] * 100
    return df

def reverse_deflate_nominal_values(df,raw_data,cpi_col_name,columns_to_deflate,target):
    """
    Inflates the deflated values in the specified columns of the dataframe using the original CPI column from raw data.

    :param df: DataFrame containing the columns to inflate and the CPI column
    :param cpi_col_name: Name of the CPI column
    :param columns_to_inflate: List of column names to inflate
    :return: DataFrame with inflated values in the specified columns
    """
    # Load the original raw data
    raw_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')

    # Ensure the CPI column is correctly aligned
    original_cpi = raw_data[cpi_col_name]

    if target in columns_to_inflate:
        df.loc[:, col] = df[col] * original_cpi / 100
    
    return df


def apply_log_transformations(df, columns_to_transform):
    for col in columns_to_transform:
        # Protect against non-positive values by applying log1p to only positive values
        df[col] = np.where(df[col] > 0, 100 * np.log1p(df[col]), df[col])
        # Handle NaNs by forward-filling and backward-filling
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)
    return df

##############################################################################3




def reverse_log_transformations(df, columns_to_transform,target):
    if target in columns_to_transform:
        # Reverse the log1p transformation applied earlier
        df = np.where(df > 0, np.expm1(df / 100), df)
    return df

def reverse_all_log_transformations(df):
    df = np.where(df > 0, np.expm1(df / 100), df)
    return df

def cap_outliers(df, cap_factor=3.0):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - cap_factor * IQR
        upper_bound = Q3 + cap_factor * IQR
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

# def reverse_cap_outliers(df,cap_factor=3.0):
#     for column in df.columns:







def reverse_best_transformations(df,raw_data,target):
    transformed_df = pd.DataFrame(index=df.index)
    raw_data = pd.read_csv('data/raw/raw_data.csv', parse_dates=True, index_col='Date')
    
    # series_data = df.ffill()  # Ensure no NaN values which might cause issues in computations
    transformation_results = pd.rea_csv('best_transformations.csv')

    # Query the CSV for the target-specific best method
    best_method_row = transformation_results[transformation_results['Target'] == target]

    if best_method_row.empty:
        raise ValueError(f"No transformation method found for target {target}")
    
    best_method = best_method_row['Best Method'].values[0]
    # Print statement to declare the column and the best transformation
    # print(f"Column: {column}, Best Method: {best_method}, ADF Statistic: {best_statistic}")
    # raw_data = raw_data[column]

    if best_method == 'Simple Differencing':
        first_value = raw_data[target_name].iloc[0]
        transformed_df[target_name] = df[target_name].cumsum() + first_value
    elif best_method == 'Rolling Mean Subtraction':
        rolling_mean = raw_data[target_name].rolling(window=7).mean()
        transformed_df[target_name] = (df[target_name] + rolling_mean).ffill()
    elif best_method == 'Rolling Mean Subtraction + Differencing':
        rolling_mean = raw_data[target_name].rolling(window=7).mean()
        first_value = raw_data[target_name].iloc[0]
        reversed_diff = df[target_name].cumsum() + first_value
        transformed_df[target_name] = (reversed_diff + rolling_mean).ffill()
    else:
        # If the method is 'None', no transformation is applied
        transformed_df[target_name] = df[target_name]

    return transformed_df

import pandas as pd
import numpy as np

def reverse_sliding_window_log(transformed_data, window_size=12):
    """
    Reverses the sliding window log transformation.
    
    Args:
    transformed_data: DataFrame or Series that has undergone sliding window log transformation.
    window_size: The size of the sliding window used during the original transformation.
    
    Returns:
    reversed_data: DataFrame or Series with the sliding window log transformation reversed.
    """
    
    # Initialize reversed_data with the same index as the transformed_data
    reversed_data = transformed_data.copy()
    
    for col in transformed_data.columns:
        for i in range(window_size - 1, len(transformed_data)):
            # Get the transformed data window
            transformed_window = transformed_data[col].iloc[i - window_size + 1: i + 1]
            
            # Reverse the log1p by applying expm1
            reversed_window = np.expm1(transformed_window)
            
            # Update the reversed_data for this window
            reversed_data[col].iloc[i - window_size + 1: i + 1] = reversed_window
        
        # Forward and backward fill for any NaNs introduced
        reversed_data[col].fillna(method='ffill', inplace=True)
        reversed_data[col].fillna(method='bfill', inplace=True)
    
    return reversed_data


def reverse_selective_logging(data, threshold):
    # Apply log1p only to values above the threshold
    data = np.where(data > threshold, np.expm1(data), data)
    # Fill NaNs to handle any missing data
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data


def reverse_relative_transform(data, target):
    raw_data = pd.read_csv('data/raw/raw_data.csv', index_col='Date', parse_dates=True)
    
    # Get the initial value of the target from the raw data
    initial_value = raw_data[target].iloc[0]
    
    # Reverse the differencing by performing cumulative sum and adding the initial value
    data = data.cumsum() + initial_value
    
    # Reverse the division by shifting the raw data and multiplying
    data = data * (raw_data[target].shift(1) + 1e-9)
    
    return data


import pandas as pd

def reverse_rolling_mean(data, window_size=5):
    # Initialize Series for reversed data
    reversed_data = pd.Series(index=data.index, dtype=float)

    for i in range(len(data)):
        # Determine the range for the window
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        # Calculate the sum of the window based on the smoothed value
        if end_idx > start_idx:
            window_mean = data.iloc[start_idx:end_idx].mean()
            # Approximate the original value by reversing the mean effect
            if i > 0:
                previous_value = reversed_data.iloc[i - 1] if not pd.isna(reversed_data.iloc[i - 1]) else data.iloc[i]
            else:
                previous_value = data.iloc[i]
            reversed_data.iloc[i] = data.iloc[i] * window_size - (window_mean * (window_size - 1))

    return reversed_data


def apply_soft_clipping(data, threshold, n=1):
    for col in data.columns:
        data[col] = data[col] / (1 + (data[col] / threshold)**n)
    return data

def reverse_soft_clipping(data, threshold,target, n=1):
    raw_data = pd.read_csv('data/raw/raw_data.csv', index_col='Date', parse_dates=True)
    data = data * (1 + (raw_data[target] / threshold) ** n)
    
    return data

import pandas as pd
import numpy as np

def reverse_deflate_nominal_values(df, raw_data, cpi_col_name, columns_to_deflate, target):
    """
    Inflates the deflated values in the specified columns of the series using the original CPI column from raw data.

    :param df: Series containing the values to inflate
    :param cpi_col_name: Name of the CPI column
    :param columns_to_deflate: List of column names to deflate
    :return: Series with inflated values
    """
    original_cpi = raw_data[cpi_col_name]
    if target in columns_to_deflate:
        df = df * original_cpi / 100
    return df

def reverse_log_transformations(df, columns_to_transform,target):
    """
    Reverses log1p transformations applied to the specified columns with overflow protection.

    :param df: DataFrame to reverse the log transformation on.
    :param columns_to_transform: List of columns to reverse.
    :return: DataFrame with log transformation reversed.
    """
    # Define a maximum value to prevent overflow
    MAX_EXP_VALUE = 700  # This prevents np.exp from causing overflow

    for col in columns_to_transform:
        # Clip the values before applying expm1 to prevent overflow
        df[col] = np.where(df[col] > 0, np.expm1(np.clip(df[col] / 100, a_min=None, a_max=MAX_EXP_VALUE)), df[col])
        # Handle NaNs by forward-filling and backward-filling (optional, based on your workflow)
        df[col].fillna(method='ffill', inplace=True)
        df[col].fillna(method='bfill', inplace=True)
    

    return df


def reverse_all_log_transformations(df):
    """
    Reverses log transformations for the entire series with overflow protection.

    :param df: Series to reverse log transformation
    :return: Series with log transformation reversed
    """
    # Define a maximum value to prevent overflow
    MAX_EXP_VALUE = 700  # Prevents np.exp from causing overflow

    # Apply the reverse log transformation with clipping to prevent overflow
    df = pd.Series(np.where(df > 0, np.expm1(np.clip(df / 100, a_min=None, a_max=MAX_EXP_VALUE)), df), index=df.index)
    
    return df


def reverse_best_transformations(df, raw_data, target):
    """
    Reverses the best transformation applied to the data based on a saved file.

    :param df: Series containing the transformed data
    :param raw_data: The original raw data
    :param target: Target column to reverse
    :return: Series with best transformation reversed
    """
    transformation_results = pd.read_csv('best_transformations.csv')
    best_method_row = transformation_results[transformation_results['Target'] == target]

    if best_method_row.empty:
        raise ValueError(f"No transformation method found for target {target}")

    best_method = best_method_row['Best Method'].values[0]

    if best_method == 'Simple Differencing':
        first_value = raw_data[target].iloc[0]
        df = df.cumsum() + first_value
    elif best_method == 'Rolling Mean Subtraction':
        rolling_mean = raw_data[target].rolling(window=7).mean()
        df = (df + rolling_mean).ffill()
    elif best_method == 'Rolling Mean Subtraction + Differencing':
        rolling_mean = raw_data[target].rolling(window=7).mean()
        first_value = raw_data[target].iloc[0]
        reversed_diff = df.cumsum() + first_value
        df = (reversed_diff + rolling_mean).ffill()
    return df

def reverse_sliding_window_log(df, window_size=12):
    """
    Reverses the sliding window log transformation.

    :param df: Series containing the transformed data
    :param window_size: The window size used during the transformation
    :return: Series with sliding window log transformation reversed
    """
    reversed_data = df.copy()
    for i in range(window_size - 1, len(df)):
        transformed_window = df.iloc[i - window_size + 1: i + 1]
        reversed_window = np.expm1(transformed_window)
        reversed_data.iloc[i - window_size + 1: i + 1] = reversed_window

    reversed_data.fillna(method='ffill', inplace=True)
    reversed_data.fillna(method='bfill', inplace=True)
    return reversed_data

def reverse_selective_logging(df, threshold):
    """
    Reverses selective logging transformation.

    :param df: Series containing the transformed data
    :param threshold: Threshold used during the transformation
    :return: Series with selective logging reversed
    """
    df = pd.Series(np.where(df > threshold, np.expm1(df), df), index=df.index)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def reverse_relative_transform(df, target):
    """
    Reverses relative transform applied to the data.

    :param df: Series containing the transformed data
    :param target: Target column to reverse
    :return: Series with relative transform reversed
    """
    raw_data = pd.read_csv('data/raw/raw_data.csv', index_col='Date', parse_dates=True)
    initial_value = raw_data[target].iloc[0]
    df = df.cumsum() + initial_value
    df = df * (raw_data[target].shift(1) + 1e-9)
    return df

def reverse_rolling_mean(df, window_size=5):
    """
    Reverses the rolling mean applied to the data.

    :param df: Series containing the transformed data
    :param window_size: Window size used during the rolling mean
    :return: Series with rolling mean reversed
    """
    reversed_data = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1

        if end_idx > start_idx:
            window_mean = df.iloc[start_idx:end_idx].mean()
            if i > 0:
                previous_value = reversed_data.iloc[i - 1] if not pd.isna(reversed_data.iloc[i - 1]) else df.iloc[i]
            else:
                previous_value = df.iloc[i]
            reversed_data.iloc[i] = df.iloc[i] * window_size - (window_mean * (window_size - 1))

    return reversed_data

def reverse_soft_clipping(df, threshold, target, n=1):
    """
    Reverses soft clipping applied to the data.

    :param df: Series containing the transformed data
    :param threshold: Threshold used during the soft clipping
    :param target: Target column to reverse
    :param n: Exponent used during the soft clipping
    :return: Series with soft clipping reversed
    """
    raw_data = pd.read_csv('data/raw/raw_data.csv', index_col='Date', parse_dates=True)
    df = df * (1 + (raw_data[target] / threshold) ** n)
    return df

import numpy as np


def safe_expm(x):
    """
    Safely compute the exponential to avoid overflow.
    Clipping the input to a maximum threshold to prevent np.exp from producing infinity.
    """
    # Maximum value for exp to avoid overflow
    MAX_EXP_VALUE = 700
    x_clipped = np.clip(x, a_min=None, a_max=MAX_EXP_VALUE)  # Clip to prevent overflow
    return np.exp(x_clipped)


def safe_cumprod(x):
    """
    Safely compute cumulative product to avoid overflow.
    Clipping the input values to prevent extremely large cumulative products.
    """
    # Maximum value for cumprod to avoid overflow
    MAX_CUMPROD_VALUE = 1e8  # Reduce the threshold to a lower, safer value
    MIN_CUMPROD_VALUE = 1e-8  # Set a reasonable lower bound to prevent underflow

    # Clip the values to prevent overflow or underflow during cumulative product
    x_clipped = np.clip(x, a_min=MIN_CUMPROD_VALUE, a_max=MAX_CUMPROD_VALUE)
    
    # Compute the cumulative product on the clipped values
    return np.cumprod(x_clipped)




#########################################################################################################

