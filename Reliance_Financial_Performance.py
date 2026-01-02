import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

#-------------------------------------------------
# 1. LOAD DATASET
#-------------------------------------------------

df = pd.read_csv(r"C:\Users\shree\Documents\RelianceRetail_financial_performance.csv")
print(df.head())

# -----------------------------------------
# STEP 2: CLEANING (ORIGINAL CODE)
# -----------------------------------------

# Convert Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Convert all numeric columns
numeric_cols = df.columns.drop('Date')
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values with column medians
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(df.head())
print("Cleaning completed.")

# -----------------------------------------
# STEP 3: FIXED & SAFE RISK SCORING MODEL
# -----------------------------------------

risk_df = df.copy()
risk_df = risk_df.sort_values(by='Date')

# 1. Calculate MoM change and Volatility

risk_df['MoM_Change'] = risk_df[numeric_cols].pct_change().mean(axis=1)
risk_df['Volatility'] = risk_df[numeric_cols].rolling(3).std().mean(axis=1)

# Replace infinity and NaN
cols = ['MoM_Change', 'Volatility']

risk_df[cols] = (
    risk_df[cols]
    .replace([np.inf, -np.inf], np.nan)  # remove infinity
    .apply(pd.to_numeric, errors='coerce')
    .fillna(0)
    .astype(float)
)

# 2. Scale MoM_Change and Volatility separately
scaler = MinMaxScaler(feature_range=(0, 100))
scaled_values = scaler.fit_transform(
    risk_df[['MoM_Change', 'Volatility']]
)

risk_df[['MoM_Change_Scaled', 'Volatility_Scaled']] = scaled_values
scaled_values = scaler.fit_transform(risk_df[['MoM_Change', 'Volatility']])
risk_df['MoM_Score'] = scaled_values[:, 0]
risk_df['Volatility_Score'] = scaled_values[:, 1]

# 3. Combine both into FINAL risk score
risk_df['Risk_Score'] = (risk_df['MoM_Score'] + risk_df['Volatility_Score']) / 2

print("Risk scoring completed successfully!")
print(risk_df[['Date', 'Risk_Score']].head())

risk_df['Risk_Level'] = pd.cut(
    risk_df['Risk_Score'],
    bins=[0, 30, 70, 100],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

# -----------------------------------------
# STEP 4: FORECASTING MODEL (PROPHET)
# -----------------------------------------

# Prophet requires columns: ds, y
forecast_input = df[['Date', numeric_cols[0]]].rename(columns={
    'Date': 'ds',
    numeric_cols[0]: 'y'
})

# Build Prophet model
model = Prophet()
model.fit(forecast_input)

# Forecast 12 future months
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast.columns = ['Date', 'Prediction', 'Lower_Bound', 'Upper_Bound']

print("\nForecasting Completed:")
print(forecast.tail())

# -----------------------------------------
# STEP 6: SAVE ALL RESULTS TO MYSQL
# -----------------------------------------

from sqlalchemy import create_engine

engine = create_engine(
    "mysql+mysqlconnector://root:root%40123@localhost/Reliance_Financial_Performance"

)

with engine.connect() as conn:
    print(" MySQL connection successful!")

#--------------------------------------------------------
# STORE RESULTS IN MYSQL
#--------------------------------------------------------

# Save cleaned data
df.to_sql("reliance_financial_cleaned", engine, if_exists="replace", index=False)
print("Uploaded → financial_cleaned")

# Save risk data (scores + levels)
risk_df.to_sql("reliance_financial_risk_scores", engine, if_exists="replace", index=False)
print("Uploaded → financial_risk_scores")

# Save forecast results
forecast.to_sql("reliance_financial_forecast", engine, if_exists="replace", index=False)
print("Uploaded → financial_forecast")

print("\n🎉 ALL DATA SUCCESSFULLY LOADED INTO MYSQL!")

#------------------------------------------------------
# EXPORT RESULT CSV TO POWER-BI
#------------------------------------------------------

output_path = r"C:\Users\shree\Documents\Financial_Performance_PowerBI"
os.makedirs(output_path, exist_ok=True)

df.to_csv(
    f"{output_path}\\Reliance_financial_cleaned.csv"
)

risk_df.to_csv(
    f"{output_path}\\Reliance_financial_risk_scores.csv"
)

forecast.to_csv(
    f"{output_path}\\Reliance_financial_forecast.csv"
)

print("🎉 CSV file saved for Power BI")