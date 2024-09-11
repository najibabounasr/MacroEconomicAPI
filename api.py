from flask import Flask, request, jsonify
import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from fetch_data import fetch_all_data
from process_data import main as process_data
from engineer_features import main as engineer_features
from train_models import main as train_models

app = Flask(__name__)

# @app.route('/fetch_data', methods=['POST'])
# def fetch_data_endpoint():
#     target_feature = request.json.get('target_feature')
#     if not target_feature:
#         return jsonify({"error": "No target feature provided"}), 400

#     try:
#         data = fetch_all_data(target_feature)
#         data.to_csv('data/raw/raw_data.csv')
#         return jsonify({"message": "Data fetched and saved."})
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 400

# @app.route('/process_data', methods=['POST'])
# def process_data_endpoint():
#     target_feature = request.json.get('target_feature')
#     if not target_feature:
#         return jsonify({"error": "No target feature provided"}), 400

#     try:
#         process_data(target_feature)
#         return jsonify({"message": "Data processed."})
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 400

from flask import Flask, request, jsonify
import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from fetch_data import fetch_all_data
from process_data import main as process_data
from engineer_features import main as engineer_features
from train_models import main as train_models

# app = Flask(__name__)

# @app.route('/engineer_features', methods=['POST'])
# def engineer_features_endpoint():
#     target_feature = request.json.get('target_feature')
#     feature_addition_rounds = int(request.json.get('feature_addition_rounds', 2))
#     feature_dropping_threshold = float(request.json.get('feature_dropping_threshold', 0.002))
#     tsfresh_fc_params = request.json.get('tsfresh_fc_params', 'MinimalFCParameters')

#     if not target_feature:
#         return jsonify({"error": "No target feature provided"}), 400

#     if tsfresh_fc_params == 'MinimalFCParameters':
#         fc_parameters = MinimalFCParameters()
#     elif tsfresh_fc_params == 'ComprehensiveFCParameters':
#         fc_parameters = ComprehensiveFCParameters()
#     elif tsfresh_fc_params == 'EfficientFCParameters':
#         fc_parameters = EfficientFCParameters()
#     else:
#         return jsonify({"error": "Invalid TSFRESH feature extraction parameters."}), 400

#     try:
#         # Load the processed data
#         X_train = pd.read_csv('data/processed/X_train_transformed.csv', index_col='Date', parse_dates=True)
#         X_test = pd.read_csv('data/processed/X_test_transformed.csv', index_col='Date', parse_dates=True)
#         y_train = pd.read_csv('data/processed/y_train_transformed.csv', index_col='Date', parse_dates=True)
#         y_test = pd.read_csv('data/processed/y_test_transformed.csv', index_col='Date', parse_dates=True)

#         engineer_features(target_feature, feature_addition_rounds, feature_dropping_threshold, fc_parameters, X_train, X_test, y_train, y_test)
#         return jsonify({"message": "Features engineered."})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/train_models', methods=['POST'])
def train_models_endpoint():
    try:
        train_params = request.json
        target_feature = train_params.get('target_feature')
        if not target_feature:
            return jsonify({'error': 'Missing target_feature parameter'}), 400

        train_models(target_feature=target_feature)

        return jsonify({'status': 'Models trained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)

