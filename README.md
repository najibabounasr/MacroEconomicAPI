# Macro Economic Forecast API

## Overview

This project provides an API for forecasting various economic indicators using machine learning. The pipeline can dynamically adjust to different target features, allowing for flexible forecasting based on the input data.

## Features

- Fetch data from FRED API
- Process and clean the data
- Engineer features
- Train various machine learning models
- Create ensemble models
- Provide a user-friendly API for making forecasts

## Installation

### Requirements

- Python 3.8+
- Flask
- pandas
- numpy
- scikit-learn
- autogluon
- catboost
- xgboost
- lightgbm
- mlflow
- dagshub
- fredapi

### Setup

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Set up DVC and remote storage (e.g., DAGsHub):

    ```sh
    dvc init
    dvc remote add -d myremote <remote-url>
    ```

## Usage

### Running the API

1. **Start the API**:

    ```sh
    python scripts/api.py
    ```

2. **Example API Requests**:

    To fetch data and initiate the pipeline, send a POST request to the `/fetch_data` endpoint with the desired target feature:

    ```sh
    curl -X POST http://127.0.0.1:5000/fetch_data -H "Content-Type: application/json" -d '{"target_feature": "GDP"}'
    ```

### Notebooks

The `notebooks` directory contains Jupyter notebooks that explain each step of the pipeline with code examples and detailed explanations.

- [01_data_fetching.ipynb](notebooks/01_data_fetching.ipynb): Fetch data from the FRED API.
- [02_data_processing.ipynb](notebooks/02_data_processing.ipynb): Process and clean the data.
- [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb): Engineer features for model training.
- [04_model_training.ipynb](notebooks/04_model_training.ipynb): Train machine learning models.
- [05_model_ensembling.ipynb](notebooks/05_model_ensembling.ipynb): Create ensemble models.

## Repository Structure

```plaintext
project-root/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── engineered/
│   └── ...
│
├── funcs/
│   ├── process_data_funcs.py
│   ├── dvc_funcs.py
│   └── ...
│
├── notebooks/
│   ├── 01_data_fetching.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_ensembling.ipynb
│
├── scripts/
│   ├── fetch_data.py
│   ├── process_data.py
│   ├── engineer_features.py
│   ├── train_models.py
│   ├── ensemble_models.py
│   └── api.py
│
├── .dvc/
│   └── config
│
├── .gitignore
├── dvc.yaml
├── README.md
├── requirements.txt
└── setup.py

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License

This project is licensed under the MIT License.
