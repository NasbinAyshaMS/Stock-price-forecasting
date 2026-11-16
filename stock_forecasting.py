import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# STOCK PRICE FORECASTING PROJECT
# ==========================================

# Stock symbol
stock_symbol = "AAPL"

print("==========================================")
print("       STOCK PRICE FORECASTING SYSTEM")
print("==========================================")
print()
print("Downloading stock data for:", stock_symbol)
print()

# Download historical stock data
data = yf.download(
    stock_symbol,
    start="2020-01-01",
    end="2026-01-01",
    auto_adjust=False
)

# Check whether data was downloaded
if data.empty:
    print("ERROR: No stock data was downloaded.")
    exit()

# Display first records
print("First 5 records:")
print(data.head())

print()
print("Last 5 records:")
print(data.tail())

print()
print("Total number of records:", len(data))

# Save data to CSV
data.to_csv("stock_data.csv")

print()
print("Stock data saved successfully as stock_data.csv")

# ==========================================
# PLOT STOCK PRICE
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    data.index,
    data["Close"],
    label="Closing Price"
)

plt.title(f"{stock_symbol} Historical Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()