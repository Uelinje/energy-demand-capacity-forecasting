PJM Energy Demand – Model Evaluation & Capacity Planning Insights
Objective
Translate forecasting performance into operational insight for short-term grid capacity planning and peak load risk assessment.

Focus Areas:

Error behavior
Peak demand performance
Risk of underprediction
Feature importance analysis
# =====================================================
# PJM Energy Demand – Full Reload + Model Reconstruction
# =====================================================
​
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
# ---------------------------
# Define features and target
# ---------------------------
​
y = df['PJM_Load_MW']
X = df.drop(columns=['PJM_Load_MW'])
​
# Chronological split
split_index = int(len(df) * 0.8)
​
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
​
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
​
# ---------------------------
# Train Random Forest
# ---------------------------
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
# Confirm objects exist
print("Reload complete.")
print("Test samples:", len(y_test))
​
# Define peak threshold (90th percentile of test demand)
peak_threshold = np.percentile(y_test, 90)
​
print("Peak threshold (MW):", round(peak_threshold, 2))
​
# Identify peak hours
peak_mask = y_test >= peak_threshold
​
print("Number of peak hours:", peak_mask.sum())
​
# Actual and predicted values for peak hours
y_test_peak = y_test[peak_mask]
y_pred_peak = y_pred_rf[peak_mask]
​
# Evaluate
mae_peak = mean_absolute_error(y_test_peak, y_pred_peak)
rmse_peak = np.sqrt(mean_squared_error(y_test_peak, y_pred_peak))
​
print("Peak MAE:", round(mae_peak, 2))
print("Peak RMSE:", round(rmse_peak, 2))
​
# Underprediction during peak hours
underprediction = y_pred_peak < y_test_peak
​
underprediction_rate = underprediction.sum() / len(y_test_peak)
​
print("Underprediction rate during peaks:", round(underprediction_rate * 100, 2), "%")
​
importances = pd.Series(rf.feature_importances_, index=X.columns)
​
importances = importances.sort_values(ascending=False)
​
importances.head(10)
​
Capacity Planning & Modeling Insights
Random Forest significantly outperforms baseline and linear regression.
Model maintains strong performance during peak demand periods.
Underprediction occurs in 57% of peak hours but magnitude remains low (1–2% of load).
Previous-hour demand (lag_1) is the dominant predictor.
Intraday seasonal encoding meaningfully improves forecasting stability.
Operational Implication: Short-term capacity planning can rely heavily on autoregressive structure, with additional seasonal adjustments improving robustness.
