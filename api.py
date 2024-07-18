from flask import Flask, request, jsonify
import pandas as pd
import os
from fetch_data import fetch_all_data
from process_data import main as process_data

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
        process_data(target_feature)
        return jsonify({"message": "Data processed."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
