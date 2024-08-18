@echo off
:: List of target features to test
set target_features=EXPGS

:: Number of feature addition rounds
set feature_addition_rounds=2

:: Feature dropping threshold
set feature_dropping_threshold=0.0002

:: TSFRESH feature extraction parameters
set tsfresh_fc_params=MinimalFCParameters

:: Loop through each target feature and make API calls
for %%t in (%target_features%) do (
  echo Testing with target feature: %%t

  @REM :: Call fetch_data endpoint
  @REM curl -X POST http://127.0.0.1:5000/fetch_data -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\"}"

  @REM :: Call process_data endpoint
  @REM curl -X POST http://127.0.0.1:5000/process_data -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\"}"

  :: Call engineer_features endpoint
  curl -X POST http://127.0.0.1:5000/engineer_features -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\", \"feature_addition_rounds\": \"%feature_addition_rounds%\", \"feature_dropping_threshold\": \"%feature_dropping_threshold%\", \"tsfresh_fc_params\": \"%tsfresh_fc_params%\"}"

  :: Call train_models endpoint
  curl -X POST http://127.0.0.1:5000/train_models -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\"}"

  echo.
)
