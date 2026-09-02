# U.S. Wind Energy Forecasting

A time-series and machine-learning analysis of monthly U.S. wind electricity generation using EIA data.

## Project Overview
This project analyzes monthly U.S. wind generation and compares classical forecasting methods with machine-learning models. The workflow includes exploratory time-series analysis, STL decomposition, Box-Cox transformation, stationarity testing, ARIMA/ETS modeling, benchmark forecasts, Random Forest, XGBoost, and scenario forecasts through 2030.

## Models
- Naive
- Seasonal Naive
- Drift
- ETS
- ARIMA
- Random Forest
- XGBoost

Models are compared using RMSE, MAE, and MAPE.

## Data
The analysis expects `eia_us_wind_monthly_2001_2025.csv` with:
- `Month`
- `Wind_Generation_Thousand_MWh`

The dataset is not included. Update `PATH_CSV` near the top of the R script to point to the dataset on your machine.

## Required R Packages
`tidyverse`, `lubridate`, `readxl`, `forecast`, `tseries`, `ranger`, and `xgboost`.

## Run
Open `wind_energy_forecasting.R` in RStudio, update `PATH_CSV`, and run the script.

## Outputs
The script writes generated results to `outputs/`, including model metrics and forecasts through 2030.

## Author
Shreya Chennapragada  
University of Oklahoma
