@echo off

:: List of target features to test
set target_features=GDP PRFI PNFI EXPGS IMPGS GCE FGCE DGS2 DGS5 DGS10

:: Loop through each target feature and make API calls
for %%t in (%target_features%) do (
  echo Testing with target feature: %%t

  :: Call fetch_data endpoint
  curl -X POST http://127.0.0.1:5000/fetch_data -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\"}"

  :: Call process_data endpoint
  curl -X POST http://127.0.0.1:5000/process_data -H "Content-Type: application/json" -d "{\"target_feature\": \"%%t\"}"

  echo.
)
