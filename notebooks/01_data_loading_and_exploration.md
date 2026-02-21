PJM Energy Demand – Data Loading & Exploration
Objective
Load hourly energy demand data and perform structured exploratory analysis to understand seasonality, trends, and variability characteristics before feature engineering and forecasting.

Scope
Validate time-series integrity
Inspect missing values
Examine distribution and seasonality
Identify structural patterns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
​
plt.style.use("seaborn-v0_8")
​
file_path = r"C:\Ujwal\RIL\Data set\Corporate process analytics\energy_demand_capacity_forecasting\PJM_Load_hourly.csv"
​
df = pd.read_csv(file_path)
​
df.head()
​
df.columns.tolist()
df['Datetime'] = pd.to_datetime(df['Datetime'])
​
df = df.sort_values('Datetime')
​
df.set_index('Datetime', inplace=True)
​
df.head()
df.index.dtype, df.shape
df.index.duplicated().sum()
# check time differences
time_diff = df.index.to_series().diff()
​
time_diff.value_counts().head()
​
# find where gap is 2 hours
gap_locations = time_diff[time_diff == pd.Timedelta(hours=2)]
​
gap_locations
​
# create complete hourly index
full_range = pd.date_range(start=df.index.min(),
                           end=df.index.max(),
                           freq='H')
​
# reindex
df = df.reindex(full_range)
​
# rename index properly
df.index.name = "Datetime"
​
# check missing after reindex
df.isna().sum()
​
# interpolate missing values
df['PJM_Load_MW'] = df['PJM_Load_MW'].interpolate(method='linear')
​
# confirm no missing values remain
df.isna().sum()
​
print("Start date:", df.index.min())
print("End date:", df.index.max())
print("Total observations:", len(df))
​
# approximate number of years
years = (df.index.max() - df.index.min()).days / 365
print("Approx years of data:", round(years, 2))
​
plt.figure(figsize=(14,5))
df['PJM_Load_MW'].plot()
plt.title("PJM Hourly Energy Demand (Full Time Range)")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.show()
​
df['Year'] = df.index.year
df['Month'] = df.index.month
​
monthly_avg = df.groupby(['Year','Month'])['PJM_Load_MW'].mean().reset_index()
​
plt.figure(figsize=(12,5))
​
for year in monthly_avg['Year'].unique():
    subset = monthly_avg[monthly_avg['Year'] == year]
    plt.plot(subset['Month'], subset['PJM_Load_MW'], label=str(year))
​
plt.legend()
plt.title("Monthly Average Load by Year")
plt.xlabel("Month")
plt.ylabel("Average Load (MW)")
plt.show()
​
df['Hour'] = df.index.hour
​
hourly_avg = df.groupby('Hour')['PJM_Load_MW'].mean()
​
plt.figure(figsize=(10,5))
hourly_avg.plot()
plt.title("Average Load by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Load (MW)")
plt.show()
​
df['DayOfWeek'] = df.index.dayofweek
​
 weekly_avg = df.groupby('DayOfWeek')['PJM_Load_MW'].mean()
​
plt.figure(figsize=(8,4))
weekly_avg.plot(kind='bar')
plt.title("Average Load by Day of Week")
plt.xlabel("Day of Week (0=Mon)")
plt.ylabel("Average Load (MW)")
plt.show()
​
Exploratory Findings Summary
The dataset spans approximately 3.75 years of hourly load data.
The time index is continuous after reindexing and interpolation.
Strong annual seasonality is present, with clear recurring monthly patterns.
Strong intraday (hourly) seasonality exists, with consistent daily load cycles.
Weekly demand variation indicates structural differences between weekdays and weekends.
Implication: Forecasting models must incorporate:

Hour-of-day effects
Day-of-week effects
Seasonal structure
Lag-based temporal dependencies
