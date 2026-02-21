PJM Energy Demand – Forecasting Models
Objective
Develop and compare structured forecasting models for 24-hour ahead energy demand prediction using engineered lag, rolling, and cyclical features.

Modeling Strategy
Baseline persistence model
Linear regression
Random forest regressor
Model comparison using MAE and RMSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
​
# ---------------------------
# Load raw data
# ---------------------------
​
file_path = r"C:\Ujwal\RIL\Data set\Corporate process analytics\energy_demand_capacity_forecasting\PJM_Load_hourly.csv"
​
df = pd.read_csv(file_path)
​
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')
df.set_index('Datetime', inplace=True)
​
# ---------------------------
# Reindex hourly & interpolate
# ---------------------------
​
full_range = pd.date_range(start=df.index.min(),
                           end=df.index.max(),
                           freq='H')
​
df = df.reindex(full_range)
df.index.name = "Datetime"
​
df['PJM_Load_MW'] = df['PJM_Load_MW'].interpolate(method='linear')
​
# ---------------------------
# Feature Engineering
# ---------------------------
​
# Lag features
df['lag_1'] = df['PJM_Load_MW'].shift(1)
df['lag_24'] = df['PJM_Load_MW'].shift(24)
df['lag_168'] = df['PJM_Load_MW'].shift(168)
​
# Rolling features
df['rolling_mean_24'] = df['PJM_Load_MW'].rolling(window=24).mean()
df['rolling_std_24'] = df['PJM_Load_MW'].rolling(window=24).std()
df['rolling_mean_168'] = df['PJM_Load_MW'].rolling(window=168).mean()
​
# Calendar features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
​
# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
​
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
​
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
​
# Drop incomplete rows
df = df.dropna()
​
df.shape
​
# Target
y = df['PJM_Load_MW']
​
# Features (exclude raw target)
feature_cols = df.columns.drop('PJM_Load_MW')
​
X = df[feature_cols]
​
X.shape
​
# Define split index (80%)
split_index = int(len(df) * 0.8)
​
# Chronological split
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
​
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
​
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
​
# Baseline prediction: use lag_1 as forecast
y_pred_baseline = X_test['lag_1']
​
# Evaluate
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
​
print("Baseline MAE:", round(mae_baseline, 2))
print("Baseline RMSE:", round(rmse_baseline, 2))
​
# Initialize model
lin_reg = LinearRegression()
​
# Fit model
lin_reg.fit(X_train, y_train)
​
# Predict
y_pred_lr = lin_reg.predict(X_test)
​
# Evaluate
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
​
print("Linear Regression MAE:", round(mae_lr, 2))
print("Linear Regression RMSE:", round(rmse_lr, 2))
​
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
​
rf.fit(X_train, y_train)
​
y_pred_rf = rf.predict(X_test)
​
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
​
print("Random Forest MAE:", round(mae_rf, 2))
print("Random Forest RMSE:", round(rmse_rf, 2))
​
# Plot last 7 days (168 hours)
hours = 168
​
plt.figure(figsize=(14,5))
​
plt.plot(y_test[-hours:].values, label="Actual")
plt.plot(y_pred_rf[-hours:], label="Random Forest Prediction")
​
plt.legend()
plt.title("Actual vs Random Forest Prediction (Last 7 Days)")
plt.show()
