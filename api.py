from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from fetch_data import fetch_all_data
from process_data import main as process_data
from engineer_features import main as engineer_features

app = Flask(__name__)

@app.route('/fetch_data', methods=['POST'])
def fetch_data_endpoint():
    target_feature = request.json.get('target_feature')
    if not target_feature:
        return jsonify({"error": "No target feature provided"}), 400

    try:
        data = fetch_all_data(target_feature)
        data.to_csv('data/raw/raw_data.csv')
        return jsonify({"message": "Data fetched and saved."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/process_data', methods=['POST'])
def process_data_endpoint():
    target_feature = request.json.get('target_feature')

    if not target_feature:
        return jsonify({"error": "No target feature provided"}), 400

    try:
        sys.argv = [sys.argv[0], f'--target={target_feature}']
        process_data(target_feature)
        return jsonify({"message": "Data processed."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/engineer_features', methods=['POST'])
def engineer_features_endpoint():
    target_feature = request.json.get('target_feature')
    feature_addition_rounds = request.json.get('feature_addition_rounds', 2)
    feature_dropping_threshold = request.json.get('feature_dropping_threshold', 0.002)
    tsfresh_fc_params = request.json.get('tsfresh_fc_params', 'MinimalFCParameters')

    if not target_feature:
        return jsonify({"error": "No target feature provided"}), 400
    
    try:
        feature_addition_rounds = int(feature_addition_rounds)
    except ValueError:
        return jsonify({"error": "Feature addition rounds must be a valid integer."}), 400

    # Ensure feature addition rounds is non-negative
    if feature_addition_rounds < 0:
        return jsonify({"error": "Feature addition rounds must be at least 0."}), 400

    # Map the string to the appropriate TSFRESH parameter object
    if tsfresh_fc_params == 'MinimalFCParameters':
        fc_parameters = MinimalFCParameters()
    elif tsfresh_fc_params == 'ComprehensiveFCParameters':
        fc_parameters = ComprehensiveFCParameters()
    elif tsfresh_fc_params == 'EfficientFCParameters':
        fc_parameters = EfficientFCParameters()
    else:
        # Default to MinimalFCParameters if the provided value is invalid
        fc_parameters = MinimalFCParameters()
        return jsonify({
            "error": "Invalid TSFRESH feature extraction parameters. Please specify one of 'MinimalFCParameters', 'ComprehensiveFCParameters', or 'EfficientFCParameters'. Defaulted to 'MinimalFCParameters'."
        }), 400

    try:
        # Log the received parameters for debugging
        print(f"Received parameters - Target: {target_feature}, Feature Addition Rounds: {feature_addition_rounds}, "
              f"Feature Dropping Threshold: {feature_dropping_threshold}, TSFRESH FC Parameters: {tsfresh_fc_params}")

        # Pass the parameters to the engineer_features function
        engineer_features(target_feature, feature_addition_rounds, feature_dropping_threshold, fc_parameters)
        return jsonify({"message": "Features engineered."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
