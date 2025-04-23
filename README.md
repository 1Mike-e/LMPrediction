# LMP & Weather-Driven Electricity Price Modeling

## 📁 Project Overview

This project focuses on analyzing and forecasting the **Locational Marginal Price (LMP)** for Michigan Hub using weather, time, and market data. It integrates historical LMP data with hourly weather observations and actual load to build predictive models.

---

## 📊 Data Sources

- **LMP Data**: `miso_LMP_21_25.csv`
- **Weather Data**: Fetched via Open-Meteo API (2021–2025)
- **Load Data**: `miso_load_act_hr.csv`

---

## 🔍 Exploratory Data Analysis

- Histograms and boxplots of LMP
- Scatter plot: LMP vs. Congestion
- Correlation matrix across weather and time variables
- Summary stats and missing value checks

---

## 🧹 Data Cleaning

- Timestamp parsing and feature extraction (hour, weekday, month)
- Removal of null values
- Outlier detection via IQR
- Merge weather and LMP data by timestamp

---

## 🛠️ Feature Engineering

- **Time-based**: `IsPeakHour`, `IsNightHour`, `Hour_sin`, `Hour_cos`
- **Weather interactions**: `temp_humidity_index`, `wind_total`
- **Flags**: `IsRaining`, `IsSnowing`
- **Lag features**: `LMP_lag_1`, `LMP_lag_24`
- **Clustering**: KMeans (n=4) on time and weather variables using PCA for visualization

---

## 🔍 Data Quality Checks

- Unique values, duplicates, timestamp issues
- Summary matrix of unique/null/consecutive changes

---

## 🤖 Modeling Approaches

### 🔹 Linear Regression

- Basic LMP prediction using congestion/loss
- Log-transform applied to target

### 🌲 Random Forest

- Full-featured and reduced models
- Feature importance ranking

### ⚡ XGBoost

- Log-transformed regression
- Forecast with load and engineered features
- RMSE and R² for evaluation

---

## 📈 Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **R² Score**: Explained variance
- **Cross-validation**: K-Fold on log-transformed targets

---

## 🔬 Final Outputs

- `df3.csv`: Merged dataset with all features
- Feature selection for reduced models
- Scatter plots: actual vs predicted
- Final models: full-reconstruction and forecast-only

---

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openmeteo-requests requests-cache retry-requests

```
 
