#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the CSV file
df = pd.read_csv("miso_LMP_21_25.csv")

# Convert to datetime if applicable
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Preview data
print("🔍 Preview of Data:")
print(df.head())

# Dataset info
print("\n📋 Dataset Info:")
print(df.info())

# Descriptive statistics
print("\n📊 Descriptive Statistics:")
print(df.describe())

# Missing values check
print("\n❗ Missing Values:")
print(df.isnull().sum())


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("miso_LMP_21_25.csv")

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Plot 1: Histogram of Michigan Hub LMP
plt.figure(figsize=(10, 4))
plt.hist(df['Michigan Hub LMP'], bins=100, edgecolor='k')
plt.title('Distribution of Michigan Hub LMP')
plt.xlabel('LMP ($/MWh)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Boxplot of Michigan Hub LMP
plt.figure(figsize=(8, 4))
plt.boxplot(df['Michigan Hub LMP'], vert=False)
plt.title('Boxplot of Michigan Hub LMP')
plt.xlabel('LMP ($/MWh)')
plt.tight_layout()
plt.show()

# Plot 3: Scatter Plot - LMP vs Congestion
plt.figure(figsize=(8, 5))
plt.scatter(df['Michigan Hub (Congestion)'], df['Michigan Hub LMP'], alpha=0.3, s=10)
plt.title('LMP vs Congestion Component')
plt.xlabel('Congestion ($/MWh)')
plt.ylabel('LMP ($/MWh)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd

# Load your data
df = pd.read_csv("miso_LMP_21_25.csv")

# Make sure the target column is clean
df = df[df['Michigan Hub LMP'].notna()]

# Step 1: Calculate Q1 and Q3
Q1 = df['Michigan Hub LMP'].quantile(0.25)
Q3 = df['Michigan Hub LMP'].quantile(0.75)
Median = df['Michigan Hub LMP'].quantile(0.50)
# Step 2: Compute IQR
IQR = Q3 - Q1

# Step 3: Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4: Flag outliers
df['is_outlier'] = (df['Michigan Hub LMP'] < lower_bound) | (df['Michigan Hub LMP'] > upper_bound)

# Step 5: Summary
outlier_count = df['is_outlier'].sum()
total_count = len(df)
outlier_percentage = (outlier_count / total_count) * 100
print(f"Q1: {Q1}")
print(f"Median: {Median}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Total records: {total_count}")
print(f"Outlier count: {outlier_count}")
print(f"Outlier percentage: {outlier_percentage:.2f}%")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox

# Load the data
df = pd.read_csv("miso_LMP_21_25.csv")

# Filter positive values only for valid transformations
df = df[df['Michigan Hub LMP'] > 0]
lmp = df['Michigan Hub LMP']

# Apply transformations
log_lmp = np.log(lmp)
sqrt_lmp = np.sqrt(lmp)
cbrt_lmp = np.cbrt(lmp)
boxcox_lmp, _ = boxcox(lmp)

# Plot the distributions
plt.figure(figsize=(14, 10))

# Original
plt.subplot(3, 2, 1)
sns.histplot(lmp, bins=100, kde=True)
plt.title("Original")

# Log
plt.subplot(3, 2, 2)
sns.histplot(log_lmp, bins=100, kde=True)
plt.title("Log Transformation")

# Square Root
plt.subplot(3, 2, 3)
sns.histplot(sqrt_lmp, bins=100, kde=True)
plt.title("Square Root Transformation")

# Cube Root
plt.subplot(3, 2, 4)
sns.histplot(cbrt_lmp, bins=100, kde=True)
plt.title("Cube Root Transformation")

# Box-Cox
plt.subplot(3, 2, 5)
sns.histplot(boxcox_lmp, bins=100, kde=True)
plt.title("Box-Cox Transformation")

plt.tight_layout()
plt.show()


# In[4]:


import pandas as pd
from statistics import mode

# Load the CSV
df = pd.read_csv("miso_LMP_21_25.csv")

# Drop missing values from the column
lmp = df['Michigan Hub LMP'].dropna()

# Calculate statistics
mean_lmp = lmp.mean()
median_lmp = lmp.median()
try:
    mode_lmp = mode(lmp)
except:
    mode_lmp = lmp.mode().tolist()  # Handles multimodal cases

# Display results
print(f"Mean: {mean_lmp:.2f}")
print(f"Median: {median_lmp:.2f}")
print(f"Mode: {mode_lmp}")


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
df = pd.read_csv("miso_LMP_21_25.csv")

# Step 2: Drop missing values and keep numeric features
df = df.dropna()
X = df[['Hour Number', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)']]
y = df['Michigan Hub LMP']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Step 7: Display results
print("Linear Regression Model")
print("------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean data
df = pd.read_csv("miso_LMP_21_25.csv")
df = df[df['Michigan Hub LMP'] > 0].dropna()  # Log requires positive values

# Step 2: Select features and target
X = df[['Hour Number', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)']]
y_log = np.log(df['Michigan Hub LMP'])  # Log-transform target

# Step 3: Split data
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Step 4: Train model
model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

# Step 5: Predict in log space and convert back
y_pred_log = model_log.predict(X_test)
y_pred = np.exp(y_pred_log)  # Inverse transform
y_test = np.exp(y_test_log)

# Step 6: Evaluate model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Step 7: Results
print("Linear Regression with Log-Transformed Target")
print("------------------------------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# In[13]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')


df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = Monday
df['Month'] = df['Timestamp'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)


# In[19]:


import pandas as pd

# Load both files
df_lmp = pd.read_csv("miso_LMP_21_25.csv")
df_temp = pd.read_csv("miso_load-temp_hr_2021.csv")
df_lmp['Timestamp'] = pd.to_datetime(df_lmp['Timestamp'], errors='coerce')
df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'], errors='coerce')


# In[17]:


df


# In[20]:


# Basic merge on exact timestamps
df_merged = pd.merge(df_lmp, df_temp, on='Timestamp', how='left')
df_merged


# In[21]:


df_temp


# In[23]:


df_merged.head(2000)


# In[25]:


get_ipython().system('pip install openmeteo-requests')
get_ipython().system('pip install requests-cache retry-requests numpy pandas')


# In[26]:


import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.7684,
	"longitude": -86.158,
	"start_date": "2021-01-01",
	"end_date": "2021-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_rain = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)


# In[27]:


hourly_dataframe.to_csv("my_tempdata.csv", index=False)


# In[28]:


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.7684,
	"longitude": -86.158,
	"start_date": "2022-01-01",
	"end_date": "2022-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_rain = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe2022 = pd.DataFrame(data = hourly_data)
print(hourly_dataframe2022)


# In[29]:


hourly_dataframe2022.to_csv("my_tempdata2022.csv", index=False)


# In[30]:


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.7684,
	"longitude": -86.158,
	"start_date": "2023-01-01",
	"end_date": "2023-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_rain = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe2023 = pd.DataFrame(data = hourly_data)
print(hourly_dataframe2023)

hourly_dataframe2023.to_csv("my_tempdata2023.csv", index=False)


# In[31]:


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.7684,
	"longitude": -86.158,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_rain = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe2024 = pd.DataFrame(data = hourly_data)
print(hourly_dataframe2024)

hourly_dataframe2024.to_csv("my_tempdata2024.csv", index=False)


# In[32]:


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.7684,
	"longitude": -86.158,
	"start_date": "2025-01-01",
	"end_date": "2025-03-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_rain = hourly.Variables(4).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(10).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe2025 = pd.DataFrame(data = hourly_data)
print(hourly_dataframe2025)

hourly_dataframe2025.to_csv("my_tempdata2025.csv", index=False)


# In[33]:


# Load both files
df_lmp = pd.read_csv("miso_LMP_21_25.csv")
df_temp = pd.read_csv("my_tempdata.csv")
df_lmp['Timestamp'] = pd.to_datetime(df_lmp['Timestamp'], errors='coerce')
df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'], errors='coerce')


# In[34]:


# Basic merge on exact timestamps
df_merged = pd.merge(df_lmp, df_temp, on='Timestamp', how='left')
df_merged


# In[36]:


df_merged['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')



df_merged['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = Monday
df_merged['Month'] = df['Timestamp'].dt.month
df_merged['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df_merged


# In[37]:


df_merged = df.drop('Hour', axis=1)
df_merged


# In[38]:


# Basic merge on exact timestamps
df_merged = pd.merge(df_lmp, df_temp, on='Timestamp', how='left')
df_merged


# In[39]:


df_merged['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')



df_merged['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0 = Monday
df_merged['Month'] = df['Timestamp'].dt.month
df_merged['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df_merged


# In[43]:


df_merged.to_csv("df_merge.csv", index=False)


# In[44]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your dataset
df = pd.read_csv("df_merge.csv")  # Replace with your file name

# Step 2: Drop rows with missing values (or fill if you prefer)
df = df.dropna()

# Step 3: Define features (X) and target (y)
feature_cols = [
    'Hour Number', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth', 'weather_code',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'DayOfWeek', 'Month', 'IsWeekend'
]

X = df[feature_cols]
y = df['Michigan Hub LMP']  # Or use np.log(y) if you want log-transformed target

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Step 8: Display results
print("Linear Regression Results")
print("-------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# In[45]:


df_merged


# In[46]:


df_merged['Timestamp'] = pd.to_datetime(df_lmp['Timestamp'], errors='coerce')
df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'], errors='coerce')
# Basic merge on exact timestamps
df1 = pd.merge(df_merged, df_temp, on='Timestamp', how='left')
df1


# In[47]:


df1.to_csv("df1.csv", index=False)


# In[48]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your dataset
df = pd.read_csv("df1.csv")  # Replace with your file name

# Step 2: Drop rows with missing values (or fill if you prefer)
df = df.dropna()

# Step 3: Define features (X) and target (y)
feature_cols = [
    'Hour Number', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth', 'weather_code',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'DayOfWeek', 'Month', 'IsWeekend'
]

X = df[feature_cols]
y = df['Michigan Hub LMP']  # Or use np.log(y) if you want log-transformed target

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Step 8: Display results
print("Linear Regression Results")
print("-------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# In[50]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("df1.csv")  # Replace with your actual file name

# Convert Timestamp to datetime (if needed)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# List of columns to include (excluding Timestamp)
columns = [
    'Hour Number', 'Michigan Hub LMP', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth',
    'weather_code', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
]

# Drop rows with missing values in selected columns
df_subset = df[columns].dropna()

# Compute correlation matrix
corr_matrix = df_subset.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Numeric and Weather Variables")
plt.tight_layout()
plt.show()


# In[51]:


df = df.drop('Hour', axis=1)
df


# In[53]:


# Load your data
df = df1

# Convert Timestamp to datetime (if needed)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# List of columns to include (excluding Timestamp)
columns = [
    'Hour Number', 'Michigan Hub LMP', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth',
    'weather_code', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
]

# Drop rows with missing values in selected columns
df_subset = df[columns].dropna()

# Compute correlation matrix
corr_matrix = df_subset.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Numeric and Weather Variables")
plt.tight_layout()
plt.show()


# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = df1

# Drop missing values (optional but recommended)
df = df.dropna()

# Define feature columns (only the ones currently available)
features = [
    'Hour Number', 'Michigan Hub LMP', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth',
    'weather_code', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
]

# Define target
X = df[features]
y = df['Michigan Hub LMP']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()


# In[56]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load data
df = df1
df = df.dropna()

# Define features and target
features = [
    'Hour Number', 'Michigan Hub LMP', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
     'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth',
    'weather_code', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
]

X = df[features]
y = df['Michigan Hub LMP']

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Run cross-validation (negative MSE, convert to RMSE)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')

# Display results
print("Cross-Validation RMSE Scores:", -cv_scores)
print("Average RMSE:", -cv_scores.mean())


# In[58]:


# Load both files

df_load = pd.read_csv("miso_load_act_hr.csv")
df_load['Timestamp'] = pd.to_datetime(df_lmp['Timestamp'], errors='coerce')


# In[60]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df_load['Timestamp'] = pd.to_datetime(df_load['Timestamp'], errors='coerce')
# Basic merge on exact timestamps
df2 = pd.merge(df, df_load, on='Timestamp', how='left')
df2


# In[61]:


import pandas as pd

# Load both files
df1 = pd.read_csv("df1.csv")
df_load1 = pd.read_csv("miso_load_act_hr.csv")
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], errors='coerce')
df_load1['Timestamp'] = pd.to_datetime(df_load1['Timestamp'], errors='coerce')


# Basic merge on exact timestamps
df3 = pd.merge(df1, df_load1, on='Timestamp', how='left')
df3


# In[62]:


df3 = df3.dropna()
df3


# In[64]:


df3 = df3.rename(columns={'LRZ 2, 7 (States: WI, MI)\nBalancing Authorities: ALTE, MGE, UPPC, WEC, WPS, MIUP, CONS, DECO Actual Load (MW)': 'Actual_load'})
df3


# In[65]:


df3.to_csv("df3.csv", index=False)


# In[72]:


get_ipython().system('pip install xgboost')
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("df3.csv")

# Log transform LMP
df['log_LMP'] = np.log(df['Michigan Hub LMP'])

# Define features
features = [
    'Hour Number', 'Michigan Hub (Congestion)', 'Michigan Hub (Loss)',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'rain', 'snowfall', 'snow_depth',
    'weather_code', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m', 'Actual_load'
]

# Drop rows with missing values
df = df[features + ['log_LMP']].dropna()

# Features and target
X = df[features]
y = df['log_LMP']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and inverse log transformation
y_pred_log = xgb_model.predict(X_test)
y_pred_actual = np.exp(y_pred_log)
y_test_actual = np.exp(y_test)

# Evaluate
rmse = mean_squared_error(y_test_actual, y_pred_actual, squared=False)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"XGBoost RMSE: {rmse:.2f}")
print(f"XGBoost R² Score: {r2:.4f}")


# In[69]:


# Load the CSV file
df = pd.read_csv("df3.csv")

# Convert to datetime if applicable
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Preview data
print("🔍 Preview of Data:")
print(df.head())

# Dataset info
print("\n📋 Dataset Info:")
print(df.info())

# Descriptive statistics
print("\n📊 Descriptive Statistics:")
print(df.describe())

# Missing values check
print("\n❗ Missing Values:")
print(df.isnull().sum())


# In[73]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df3 = pd.read_csv("df3.csv")  # Update path if needed

# 1. Overview: structure and data types
print("📋 Data Info:")
print(df3.info())

# 2. Missing values
print("\n❗ Missing Values:")
print(df3.isnull().sum())

# 3. Summary statistics
print("\n📊 Summary Statistics:")
print(df3.describe())

# 4. Distribution plot of Michigan Hub LMP
plt.figure(figsize=(10, 4))
sns.histplot(df3['Michigan Hub LMP'], bins=100, kde=True)
plt.title("Distribution of Michigan Hub LMP")
plt.xlabel("LMP ($/MWh)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Correlation matrix heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df3.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()


# In[74]:


import pandas as pd
import numpy as np

# Load data
df3 = pd.read_csv("df3.csv")  # Replace with your actual path if needed

# ----------------------------
# 1. Log-transform LMP
# ----------------------------
df3['log_LMP'] = np.log(df3['Michigan Hub LMP'])

# ----------------------------
# 2. Detect outliers using IQR
# ----------------------------
Q1 = df3['Michigan Hub LMP'].quantile(0.25)
Q3 = df3['Michigan Hub LMP'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag rows as outliers
df3['is_outlier'] = (df3['Michigan Hub LMP'] < lower_bound) | (df3['Michigan Hub LMP'] > upper_bound)

# Print summary
outlier_count = df3['is_outlier'].sum()
total_rows = len(df3)
outlier_pct = (outlier_count / total_rows) * 100

print(f"Total Rows: {total_rows}")
print(f"Outliers Detected: {outlier_count} ({outlier_pct:.2f}%)")
print(f"LMP Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# ----------------------------
# 3. (Optional) Remove Outliers
# ----------------------------
# df3 = df3[~df3['is_outlier']]

# ----------------------------
# 4. View distribution of log_LMP
# ----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))
sns.histplot(df3['log_LMP'], bins=100, kde=True)
plt.title("Distribution of log-transformed Michigan Hub LMP")
plt.xlabel("log(LMP)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.grid(True)
plt.show()


# In[76]:


df3


# In[77]:


df3 = df.drop('Hour', axis=1)
df3


# In[78]:


# 1. Number of unique values per column
unique_counts = df3.nunique().sort_values(ascending=False)
print("🔍 Unique value counts per column:")
print(unique_counts)

# 2. Check for duplicate rows
duplicates = df3.duplicated().sum()
print(f"\n🧼 Duplicate rows: {duplicates}")

# 3. Check for duplicate timestamps (if present)
if 'Timestamp' in df3.columns:
    timestamp_dupes = df3['Timestamp'].duplicated().sum()
    print(f"🕒 Duplicate timestamps: {timestamp_dupes}")


# In[79]:


# 1. Unique value count per column
unique_counts = df3.nunique().sort_values(ascending=False)

# 2. Null value count per column
null_counts = df3.isnull().sum().sort_values(ascending=False)

# 3. Consecutive rule: Count how many values change from one row to the next
consecutive_change_counts = df3.ffill().ne(df3.shift()).sum().sort_values(ascending=False)

# 4. Combine into one summary DataFrame
quality_checks = pd.DataFrame({
    'Unique Values': unique_counts,
    'Null Count': null_counts,
    'Consecutive Changes': consecutive_change_counts
})

# 5. Display results
print("🔍 Data Quality Summary:")
print(quality_checks)


# In[80]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df3 = pd.read_csv("df3.csv")

# Compute Pearson correlation with LMP
corr_matrix = df3.corr(numeric_only=True)
top_corr = corr_matrix['Michigan Hub LMP'].abs().sort_values(ascending=False)

# Exclude LMP itself and select top 5 correlated features
top_features = top_corr.drop('Michigan Hub LMP').head(5).index.tolist()

# Plot scatter plots for top 5 correlated features vs LMP
plt.figure(figsize=(16, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=df3, x=feature, y='Michigan Hub LMP', alpha=0.3)
    plt.title(f"LMP vs {feature}")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Scatter Plots of Top Correlated Features with LMP", fontsize=16, y=1.03)
plt.show()


# In[81]:


import pandas as pd
import numpy as np

# Load the dataset
df3 = pd.read_csv("df3.csv")

# ---- TIME-BASED FEATURES ----
# Flag peak and night hours
df3['IsPeakHour'] = df3['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df3['IsNightHour'] = df3['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

# ---- WEATHER INTERACTIONS ----
# Combined temperature-humidity effect
df3['temp_humidity_index'] = df3['temperature_2m'] * df3['relative_humidity_2m']

# Total wind strength
df3['wind_total'] = df3['wind_speed_10m'] + df3['wind_gusts_10m']

# ---- PRECIPITATION FLAGS ----
df3['IsSnowing'] = (df3['snowfall'] > 0).astype(int)
df3['IsRaining'] = (df3['rain'] > 0).astype(int)

# ---- CYCLICAL ENCODING FOR HOUR ----
df3['Hour_sin'] = np.sin(2 * np.pi * df3['Hour'] / 24)
df3['Hour_cos'] = np.cos(2 * np.pi * df3['Hour'] / 24)

# ---- LAG FEATURES ----
df3['LMP_lag_1'] = df3['Michigan Hub LMP'].shift(1)
df3['LMP_lag_24'] = df3['Michigan Hub LMP'].shift(24)

# ---- DROP NA VALUES FROM LAGS ----
df3 = df3.dropna()

# Optional: View new feature summary
print(df3[[
    'IsPeakHour', 'IsNightHour', 'temp_humidity_index', 'wind_total',
    'IsSnowing', 'IsRaining', 'Hour_sin', 'Hour_cos',
    'LMP_lag_1', 'LMP_lag_24'
]].describe())


# In[86]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df3 = pd.read_csv("df3.csv")  # Update path if needed

# Recalculate engineered features (if not already done)
df3['wind_total'] = df3['wind_speed_10m'] + df3['wind_gusts_10m']
df3['IsPeakHour'] = df3['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df3['IsNightHour'] = df3['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df3['IsSnowing'] = (df3['snowfall'] > 0).astype(int)
df3['IsRaining'] = (df3['rain'] > 0).astype(int)
df3['Hour_sin'] = np.sin(2 * np.pi * df3['Hour'] / 24)
df3['Hour_cos'] = np.cos(2 * np.pi * df3['Hour'] / 24)

# ---------------------------
# Select features for clustering
# ---------------------------
clustering_features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos'
]

# Standardize features
X = df3[clustering_features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# KMeans Clustering
# ---------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df3.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------------------
# PCA for Visualization
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df3.loc[X.index, 'PCA1'] = X_pca[:, 0]
df3.loc[X.index, 'PCA2'] = X_pca[:, 1]

# ---------------------------
# Plot Clusters
# ---------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df3.loc[X.index], x='PCA1', y='PCA2', hue='Cluster', palette='tab10', alpha=0.6)
plt.title("KMeans Clustering (PCA View of Weather & Time Features)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[83]:


df3


# In[87]:


df3


# In[88]:


df3.to_csv("df3.csv", index=False)


# In[92]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load your dataset
df3 = pd.read_csv("df3.csv")  # Update path as needed

# --- Feature Engineering ---
df3['wind_total'] = df3['wind_speed_10m'] + df3['wind_gusts_10m']
df3['IsPeakHour'] = df3['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df3['IsNightHour'] = df3['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df3['IsSnowing'] = (df3['snowfall'] > 0).astype(int)
df3['IsRaining'] = (df3['rain'] > 0).astype(int)
df3['Hour_sin'] = np.sin(2 * np.pi * df3['Hour'] / 24)
df3['Hour_cos'] = np.cos(2 * np.pi * df3['Hour'] / 24)
df3['LMP_lag_1'] = df3['Michigan Hub LMP'].shift(1)
df3['LMP_lag_24'] = df3['Michigan Hub LMP'].shift(24)
df3['log_LMP'] = np.log(df3['Michigan Hub LMP'])

# Optional: Cluster (if already computed, skip this step)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
cluster_features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos'
]
df3.dropna(subset=cluster_features, inplace=True)
X_cluster = StandardScaler().fit_transform(df3[cluster_features])
df3['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(X_cluster)

# --- Define model features ---
features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos', 'LMP_lag_1', 'LMP_lag_24',
    'Cluster', 'Actual_load'
]

# Drop NA rows
df_model = df3[features + ['log_LMP']].dropna()
X = df_model[features]
y = df_model['log_LMP']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Regressor ---
xgb_model = XGBRegressor(
    n_estimators=50,        # reduce for speed
    learning_rate=0.1,
    max_depth=4,            # balance accuracy/speed
    random_state=42
)
xgb_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred_log = xgb_model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_actual = np.exp(y_test)

rmse = mean_squared_error(y_actual, y_pred, squared=False)
r2 = r2_score(y_actual, y_pred)

print(f"XGBoost RMSE: ${rmse:.2f}/MWh")
print(f"XGBoost R² Score: {r2:.4f}")


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- Load dataset ---
df3 = pd.read_csv("df3.csv")  # Adjust path if needed

# --- Feature Engineering ---
df3['wind_total'] = df3['wind_speed_10m'] + df3['wind_gusts_10m']
df3['IsPeakHour'] = df3['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df3['IsNightHour'] = df3['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df3['IsSnowing'] = (df3['snowfall'] > 0).astype(int)
df3['IsRaining'] = (df3['rain'] > 0).astype(int)
df3['Hour_sin'] = np.sin(2 * np.pi * df3['Hour'] / 24)
df3['Hour_cos'] = np.cos(2 * np.pi * df3['Hour'] / 24)
df3['LMP_lag_1'] = df3['Michigan Hub LMP'].shift(1)
df3['LMP_lag_24'] = df3['Michigan Hub LMP'].shift(24)
df3['log_LMP'] = np.log(df3['Michigan Hub LMP'])

# --- Select features and target ---
features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos', 'LMP_lag_1', 'LMP_lag_24',
    'Actual_load'
]
df_model = df3[features + ['log_LMP']].dropna()

X = df_model[features]
y = df_model['log_LMP']

# --- Split the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

# --- Train XGBoost ---
xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)

# --- Plot Side-by-Side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

xgb_importances.plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('XGBoost Feature Importance')
axes[0].set_xlabel('Importance')

rf_importances.plot(kind='barh', ax=axes[1], color='orange')
axes[1].set_title('Random Forest Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Load dataset ---
df = pd.read_csv("df3.csv")  # Adjust path if needed

# --- Feature Engineering ---
df['wind_total'] = df['wind_speed_10m'] + df['wind_gusts_10m']
df['IsPeakHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df['IsNightHour'] = df['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df['IsSnowing'] = (df['snowfall'] > 0).astype(int)
df['IsRaining'] = (df['rain'] > 0).astype(int)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['LMP_lag_1'] = df['Michigan Hub LMP'].shift(1)
df['LMP_lag_24'] = df['Michigan Hub LMP'].shift(24)
df['log_LMP'] = np.log(df['Michigan Hub LMP'])

# --- Define feature sets ---
base_features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos', 'LMP_lag_1', 'LMP_lag_24', 'Actual_load'
]

full_features = base_features + ['Michigan Hub (Congestion)', 'Michigan Hub (Loss)']

# === MODEL 1: Forecast-Only ===
df_model1 = df[base_features + ['log_LMP']].dropna()
X1 = df_model1[base_features]
y1 = df_model1['log_LMP']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model1.fit(X1_train, y1_train)

y1_pred_log = model1.predict(X1_test)
y1_pred = np.exp(y1_pred_log)
y1_actual = np.exp(y1_test)

mse1 = mean_squared_error(y1_actual, y1_pred)
rmse1 = np.sqrt(mse1)
r2_1 = r2_score(y1_actual, y1_pred)

# === MODEL 2: Full Reconstruction ===
df_model2 = df[full_features + ['log_LMP']].dropna()
X2 = df_model2[full_features]
y2 = df_model2['log_LMP']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model2.fit(X2_train, y2_train)

y2_pred_log = model2.predict(X2_test)
y2_pred = np.exp(y2_pred_log)
y2_actual = np.exp(y2_test)

mse2 = mean_squared_error(y2_actual, y2_pred)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y2_actual, y2_pred)

# --- Print Results ---
print("🔹 Model 1: Forecast-Only (No Congestion or Loss)")
print(f"RMSE: ${rmse1:.2f}/MWh")
print(f"R² Score: {r2_1:.4f}\n")

print("🔸 Model 2: Full Reconstruction (With Congestion + Loss)")
print(f"RMSE: ${rmse2:.2f}/MWh")
print(f"R² Score: {r2_2:.4f}")


# In[4]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# Define RMSE scorer (compatible with ArcGIS)
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred))), greater_is_better=False)

# Use log_LMP as target for numerical stability
X_cv = df_model1[base_features]
y_cv = df_model1['log_LMP']

# Initialize model and cross-validation
cv_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation (note: negative values mean lower RMSE is better)
cv_rmse_scores = cross_val_score(cv_model, X_cv, y_cv, scoring=rmse_scorer, cv=cv)

# Convert to positive RMSE
cv_rmse_scores = -cv_rmse_scores
print(f"Cross-Validation RMSE (log LMP): {cv_rmse_scores}")
print(f"Average RMSE: {np.mean(cv_rmse_scores):.2f}")


# In[5]:


import pandas as pd

# Train model on full set
model1.fit(X1_train, y1_train)
importances = pd.Series(model1.feature_importances_, index=X1.columns)

# Drop features with importance below threshold (e.g. 1%)
threshold = 0.01
selected_features = importances[importances > threshold].index.tolist()

print("Selected Features:", selected_features)


# In[6]:


X_sel = df_model1[selected_features]
y_sel = df_model1['log_LMP']

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_sel, y_sel, test_size=0.2, random_state=42)

model_sel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_sel.fit(X_train_sel, y_train_sel)

y_pred_sel_log = model_sel.predict(X_test_sel)
y_pred_sel = np.exp(y_pred_sel_log)
y_actual_sel = np.exp(y_test_sel)

# Manual RMSE + R2
rmse_sel = np.sqrt(mean_squared_error(y_actual_sel, y_pred_sel))
r2_sel = r2_score(y_actual_sel, y_pred_sel)

print(f"Reduced Model RMSE: ${rmse_sel:.2f}/MWh")
print(f"Reduced Model R²: {r2_sel:.4f}")


# In[7]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_actual_sel, y_pred_sel, alpha=0.4, color='royalblue')
plt.plot([min(y_actual_sel), max(y_actual_sel)], [min(y_actual_sel), max(y_actual_sel)], 'r--', label='Ideal Prediction')
plt.xlabel('Actual LMP ($/MWh)')
plt.ylabel('Predicted LMP ($/MWh)')
plt.title('Predicted vs Actual LMP (Reduced Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# Define RMSE scorer (compatible with ArcGIS)
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred))), greater_is_better=False)

# Use log_LMP as target for numerical stability
X_cv = df_model2[base_features]
y_cv = df_model2['log_LMP']

# Initialize model and cross-validation
cv_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation (note: negative values mean lower RMSE is better)
cv_rmse_scores = cross_val_score(cv_model, X_cv, y_cv, scoring=rmse_scorer, cv=cv)

# Convert to positive RMSE
cv_rmse_scores = -cv_rmse_scores
print(f"Cross-Validation RMSE (log LMP): {cv_rmse_scores}")
print(f"Average RMSE: {np.mean(cv_rmse_scores):.2f}")


# In[13]:


import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 42.3314,
	"longitude": -83.0457,
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "snowfall"],
	"wind_speed_unit": "mph",
	"temperature_unit": "fahrenheit",
	"precipitation_unit": "inch",
	"start_date": "2025-04-01",
	"end_date": "2025-04-13"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["snowfall"] = hourly_snowfall

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)


# In[15]:


hourly_dataframe.to_csv("hourlytemp1.csv", index=False)


# In[16]:


df3


# In[17]:


df3.to_csv("lag1.csv", index=False)


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

# Example: predicted and actual values from Model 2
# Replace these with your variables if already computed:
# y2_pred_log = model2.predict(X2_test)
# y2_pred = np.exp(y2_pred_log)
# y2_actual = np.exp(y2_test)

# Sort by actual timestamp index (optional)
actual_vs_pred_df = pd.DataFrame({
    'Actual LMP': y2_actual,
    'Predicted LMP': y2_pred
})
actual_vs_pred_df = actual_vs_pred_df.reset_index(drop=True)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual_vs_pred_df['Actual LMP'], label='Actual LMP', color='black', linewidth=2)
plt.plot(actual_vs_pred_df['Predicted LMP'], label='Predicted LMP (Model 2)', color='blue', linestyle='--')
plt.title("Model 2: Full Reconstruction\nPredicted vs Actual LMP")
plt.xlabel("Sample Index (Hour)")
plt.ylabel("LMP ($/MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Load dataset ---
df = pd.read_csv("df3.csv")  # Adjust path if needed

# --- Feature Engineering ---
df['wind_total'] = df['wind_speed_10m'] + df['wind_gusts_10m']
df['IsPeakHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df['IsNightHour'] = df['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df['IsSnowing'] = (df['snowfall'] > 0).astype(int)
df['IsRaining'] = (df['rain'] > 0).astype(int)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['LMP_lag_1'] = df['Michigan Hub LMP'].shift(1)
df['LMP_lag_24'] = df['Michigan Hub LMP'].shift(24)
df['log_LMP'] = np.log(df['Michigan Hub LMP'])

# --- Define feature sets ---
base_features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total',
    'IsPeakHour', 'IsNightHour', 'IsSnowing', 'IsRaining',
    'Hour_sin', 'Hour_cos', 'LMP_lag_1', 'LMP_lag_24', 'Actual_load'
]

full_features = base_features + ['Michigan Hub (Congestion)', 'Michigan Hub (Loss)']

# === MODEL 1: Forecast-Only ===
df_model1 = df[base_features + ['log_LMP']].dropna()
X1 = df_model1[base_features]
y1 = df_model1['log_LMP']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model1.fit(X1_train, y1_train)

y1_pred_log = model1.predict(X1_test)
y1_pred = np.exp(y1_pred_log)
y1_actual = np.exp(y1_test)

mse1 = mean_squared_error(y1_actual, y1_pred)
rmse1 = np.sqrt(mse1)
r2_1 = r2_score(y1_actual, y1_pred)

# === MODEL 2: Full Reconstruction ===
df_model2 = df[full_features + ['log_LMP']].dropna()
X2 = df_model2[full_features]
y2 = df_model2['log_LMP']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model2.fit(X2_train, y2_train)

y2_pred_log = model2.predict(X2_test)
y2_pred = np.exp(y2_pred_log)
y2_actual = np.exp(y2_test)

mse2 = mean_squared_error(y2_actual, y2_pred)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y2_actual, y2_pred)

# --- Print Results ---
print("🔹 Model 1: Forecast-Only (No Congestion or Loss)")
print(f"RMSE: ${rmse1:.2f}/MWh")
print(f"R² Score: {r2_1:.4f}\n")

print("🔸 Model 2: Full Reconstruction (With Congestion + Loss)")
print(f"RMSE: ${rmse2:.2f}/MWh")
print(f"R² Score: {r2_2:.4f}")


# In[26]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare your dataset
df = pd.read_csv("df3.csv")

# --- Feature Engineering ---
df['wind_total'] = df['wind_speed_10m'] + df['wind_gusts_10m']
df['IsPeakHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df['IsNightHour'] = df['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['LMP_lag_1'] = df['Michigan Hub LMP'].shift(1)
df['LMP_lag_24'] = df['Michigan Hub LMP'].shift(24)
df['log_LMP'] = np.log(df['Michigan Hub LMP'])

# Create future targets
for i in range(1, 25):
    df[f'log_LMP_t+{i}'] = df['log_LMP'].shift(-i)

# Define features
features = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'snowfall', 'wind_total', 'Actual_load',
    'LMP_lag_1', 'LMP_lag_24', 'Hour_sin', 'Hour_cos',
    'IsPeakHour', 'IsNightHour'
]

# Drop rows with NaNs
df_clean = df[features + [f'log_LMP_t+{i}' for i in range(1, 25)]].dropna()

# Train 6 models (t+1 to t+6)
models = {}
results = []

for i in range(1, 25):
    target = f'log_LMP_t+{i}'
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = np.exp(model.predict(X_test))
    y_actual = np.exp(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    models[f't+{i}'] = model
    results.append((f't+{i}', rmse, r2))

# Print results
print("🔍 Forecast Performance (XGBoost):")
for step, rmse, r2 in results:
    print(f"{step}: RMSE = ${rmse:.2f}/MWh | R² = {r2:.4f}")


# In[ ]:




