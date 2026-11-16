import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading stock data...")

# Load data
data = pd.read_csv("stock_data.csv", skiprows=[1])

print("Data loaded successfully!")

# Convert date
data["Date"] = pd.to_datetime(data["Price"], errors="coerce")

# Convert numerical columns
columns = ["Close", "High", "Low", "Open", "Volume"]

for column in columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove missing values
data = data.dropna()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

data["Previous_Close"] = data["Close"].shift(1)

data["MA_5"] = data["Close"].rolling(window=5).mean()

data["MA_20"] = data["Close"].rolling(window=20).mean()

data = data.dropna()

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = data[
    [
        "Previous_Close",
        "MA_5",
        "MA_20",
        "Open",
        "High",
        "Low",
        "Volume"
    ]
]

y = data["Close"]

# ---------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ---------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest training completed!")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========== RANDOM FOREST RESULTS ==========")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
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

next_prediction = model.predict(next_day_data)[0]

print("\n===================================")
print("Random Forest Next Day Prediction:",
      round(next_prediction, 2))
print("===================================")

# Save results
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

results.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nPrediction results saved as random_forest_predictions.csv")