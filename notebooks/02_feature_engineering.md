PJM Energy Demand – Feature Engineering
Objective
Engineer time-series features that capture temporal dependencies, seasonality, and load dynamics to support robust short-term forecasting.

Feature Categories
Lag Features
Rolling Statistics
Calendar Features
Cyclical Encoding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
file_path = r"C:\Ujwal\RIL\Data set\Corporate process analytics\energy_demand_capacity_forecasting\PJM_Load_hourly.csv"
​
df = pd.read_csv(file_path)
​
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')
df.set_index('Datetime', inplace=True)
​
# reindex hourly
full_range = pd.date_range(start=df.index.min(),
                           end=df.index.max(),
                           freq='H')
​
df = df.reindex(full_range)
df.index.name = "Datetime"
​
# interpolate missing
df['PJM_Load_MW'] = df['PJM_Load_MW'].interpolate(method='linear')
​
df.head()
​
# 1 hour lag
df['lag_1'] = df['PJM_Load_MW'].shift(1)
​
# 24 hour lag (same hour previous day)
df['lag_24'] = df['PJM_Load_MW'].shift(24)
​
# 168 hour lag (same hour previous week)
df['lag_168'] = df['PJM_Load_MW'].shift(168)
​
df[['PJM_Load_MW', 'lag_1', 'lag_24', 'lag_168']].head(200)
​
# 24-hour rolling mean
df['rolling_mean_24'] = df['PJM_Load_MW'].rolling(window=24).mean()
​
# 24-hour rolling std
df['rolling_std_24'] = df['PJM_Load_MW'].rolling(window=24).std()
​
# 7-day rolling mean (168 hours)
df['rolling_mean_168'] = df['PJM_Load_MW'].rolling(window=168).mean()
​
df[['PJM_Load_MW', 'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']].head(200)
​
# Calendar features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
​
df[['hour', 'day_of_week', 'month', 'year']].head()
​
# Hour cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
​
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
​
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
​
df.isna().sum()
​
df = df.dropna()
​
df.shape
​
Feature Engineering Summary
The following feature categories were engineered:

Lag Features:

1-hour lag
24-hour lag
168-hour lag (weekly memory)
Rolling Statistics:

24-hour rolling mean
24-hour rolling standard deviation
168-hour rolling mean
Calendar Features:

Hour
Day of Week
Month
Year
Cyclical Encoding:

Hour (sin/cos)
Day of Week (sin/cos)
Month (sin/cos)
All rows with incomplete lag or rolling data were removed to prevent leakage and ensure valid supervised learning setup.

​
