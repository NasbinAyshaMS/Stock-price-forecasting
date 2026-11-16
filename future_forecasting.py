import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# SELECT STOCK
# ============================================

stock_symbol = "AAPL"

print("Downloading stock data...")

data = yf.download(
    stock_symbol,
    start="2020-01-01",
    auto_adjust=False,
    progress=False
)

if data.empty:
    print("No stock data found.")
    exit()


# Handle yfinance MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


data.reset_index(inplace=True)


# Convert Close to numeric
data["Close"] = pd.to_numeric(
    data["Close"],
    errors="coerce"
)

data.dropna(inplace=True)


# ============================================
# FEATURE ENGINEERING
# ============================================

data["Previous_Close"] = data["Close"].shift(1)

data["MA_5"] = (
    data["Close"]
    .rolling(5)
    .mean()
)

data["MA_20"] = (
    data["Close"]
    .rolling(20)
    .mean()
)

data.dropna(inplace=True)


# ============================================
# TRAIN MODEL
# ============================================

features = [
    "Previous_Close",
    "MA_5",
    "MA_20"
]

X = data[features]
y = data["Close"]


model = LinearRegression()

model.fit(X, y)

print("Model trained successfully!")


# ============================================
# FUTURE FORECAST
# ============================================

forecast_days = 7

history = list(data["Close"])

future_prices = []


for day in range(1, forecast_days + 1):

    previous_close = history[-1]

    ma_5 = np.mean(history[-5:])

    ma_20 = np.mean(history[-20:])

    input_data = pd.DataFrame({
        "Previous_Close": [previous_close],
        "MA_5": [ma_5],
        "MA_20": [ma_20]
    })

    prediction = model.predict(input_data)[0]

    future_prices.append(prediction)

    history.append(prediction)


# ============================================
# DISPLAY FORECAST
# ============================================

print("\n===================================")
print("7-DAY STOCK PRICE FORECAST")
print("===================================")

for day, price in enumerate(
    future_prices,
    start=1
):

    print(
        f"Day {day}: ${price:.2f}"
    )


# ============================================
# SAVE FORECAST
# ============================================

forecast_data = pd.DataFrame({
    "Day": range(1, forecast_days + 1),
    "Predicted Price": future_prices
})

forecast_data.to_csv(
    "future_forecast.csv",
    index=False
)

print(
    "\nForecast saved as future_forecast.csv"
)


# ============================================
# GRAPH
# ============================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, forecast_days + 1),
    future_prices,
    marker="o"
)

plt.title(
    f"{stock_symbol} - 7 Day Future Price Forecast"
)

plt.xlabel("Future Day")

plt.ylabel("Predicted Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "future_forecast.png"
)

plt.show()

print(
    "Forecast graph saved as future_forecast.png"
)