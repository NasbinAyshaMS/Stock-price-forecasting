
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading stock data...")

# Read CSV and skip the extra Ticker row
data = pd.read_csv("stock_data.csv", skiprows=[1])

print("Data loaded successfully!")
print("Number of records:", len(data))

# Display column names
print("\nColumns:")
print(data.columns.tolist())

# Convert Date column
data["Date"] = pd.to_datetime(data["Price"], errors="coerce")

# Remove invalid rows
data = data.dropna(subset=["Date"])

# Rename the actual stock price columns
data = data.rename(columns={
    "Close": "Close",
    "High": "High",
    "Low": "Low",
    "Open": "Open",
    "Volume": "Volume"
})

# Convert price columns to numeric
for column in ["Close", "High", "Low", "Open", "Volume"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove missing values
data = data.dropna()

print("\nCleaned data:")
print(data.head())

print("\nNumber of records after cleaning:", len(data))

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

# Previous day's closing price
data["Previous_Close"] = data["Close"].shift(1)

# 5-day moving average
data["MA_5"] = data["Close"].rolling(window=5).mean()

# 20-day moving average
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Remove rows with missing values
data = data.dropna()

# ---------------------------------------------------
# INPUT AND OUTPUT
# ---------------------------------------------------

X = data[[
    "Previous_Close",
    "MA_5",
    "MA_20",
    "Open",
    "High",
    "Low",
    "Volume"
]]

y = data["Close"]

# ---------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------

# Use chronological split for stock time-series data
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ---------------------------------------------------
# LINEAR REGRESSION MODEL
# ---------------------------------------------------

print("\nTraining Linear Regression model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 4))

# ---------------------------------------------------
# NEXT DAY PREDICTION
# ---------------------------------------------------

latest = data.iloc[-1]

next_day_data = pd.DataFrame({
    "Previous_Close": [latest["Close"]],
    "MA_5": [latest["MA_5"]],
    "MA_20": [latest["MA_20"]],
    "Open": [latest["Open"]],
    "High": [latest["High"]],
    "Low": [latest["Low"]],
    "Volume": [latest["Volume"]]
})

next_day_prediction = model.predict(next_day_data)[0]

print("\n===================================")
print("Predicted Next Closing Price:", round(next_day_prediction, 2))
print("===================================")

# ---------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

results.to_csv("prediction_results.csv", index=False)

print("\nPrediction results saved as prediction_results.csv")

