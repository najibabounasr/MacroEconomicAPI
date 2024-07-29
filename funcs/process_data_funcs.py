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
# Function to apply logarithmic transformation
def apply_log_transformations(df, columns_to_transform):
    for col in columns_to_transform:
        df[col] = 100 * np.log(df[col])
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
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
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
        print(f"Column: {column}, Best Method: {best_method}, ADF Statistic: {best_statistic}")
        
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

#########################################################################################################

