import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# GLOBAL VARIABLES
# ============================================================

stock_data = None
actual_prices = None
predicted_prices = None
future_prices = None
model = None


# ============================================================
# DOWNLOAD AND PREPARE STOCK DATA
# ============================================================

def get_stock_data(symbol):

    data = yf.download(
        symbol,
        start="2020-01-01",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data found for {symbol}"
        )

    # Handle yfinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] for column in data.columns
        ]

    data.reset_index(inplace=True)

    # Make sure required columns exist
    required_columns = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume"
    ]

    for column in required_columns:

        if column not in data.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data.dropna(inplace=True)

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # Previous closing price
    data["Previous_Close"] = (
        data["Close"].shift(1)
    )

    # 5-day moving average
    data["MA_5"] = (
        data["Close"]
        .rolling(window=5)
        .mean()
    )

    # 20-day moving average
    data["MA_20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data.dropna(inplace=True)

    return data


# ============================================================
# LOAD DATA AND TRAIN MODEL
# ============================================================

def load_data_and_predict():

    global stock_data
    global actual_prices
    global predicted_prices
    global model

    symbol = stock_var.get()

    if not symbol:

        messagebox.showerror(
            "Error",
            "Please select a stock."
        )

        return

    try:

        status_label.config(
            text=f"Downloading {symbol} data..."
        )

        window.update_idletasks()

        # Download data
        data = get_stock_data(symbol)

        # Create features
        data = create_features(data)

        if len(data) < 100:

            raise ValueError(
                "Not enough stock data for training."
            )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        features = [
            "Previous_Close",
            "MA_5",
            "MA_20",
            "Open",
            "High",
            "Low",
            "Volume"
        ]

        X = data[features]

        y = data["Close"]

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        split_index = int(
            len(data) * 0.8
        )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # ----------------------------------------------------
        # LINEAR REGRESSION
        # ----------------------------------------------------

        status_label.config(
            text="Training Linear Regression model..."
        )

        window.update_idletasks()

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # TEST PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # MODEL METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # NEXT PRICE PREDICTION
        # ----------------------------------------------------

        latest = data.iloc[-1]

        latest_features = pd.DataFrame({

            "Previous_Close": [
                latest["Close"]
            ],

            "MA_5": [
                latest["MA_5"]
            ],

            "MA_20": [
                latest["MA_20"]
            ],

            "Open": [
                latest["Open"]
            ],

            "High": [
                latest["High"]
            ],

            "Low": [
                latest["Low"]
            ],

            "Volume": [
                latest["Volume"]
            ]

        })

        next_price = model.predict(
            latest_features
        )[0]

        current_price = float(
            latest["Close"]
        )

        # ----------------------------------------------------
        # UPDATE GUI
        # ----------------------------------------------------

        current_price_label.config(
            text=f"Current Price: {current_price:.2f}"
        )

        predicted_price_label.config(
            text=f"Predicted Price: {next_price:.2f}"
        )

        mae_label.config(
            text=f"MAE: {mae:.2f}"
        )

        rmse_label.config(
            text=f"RMSE: {rmse:.2f}"
        )

        r2_label.config(
            text=f"R² Score: {r2:.4f}"
        )

        if next_price > current_price:

            prediction_label.config(
                text="Prediction: UP",
                foreground="green"
            )

        elif next_price < current_price:

            prediction_label.config(
                text="Prediction: DOWN",
                foreground="red"
            )

        else:

            prediction_label.config(
                text="Prediction: STABLE",
                foreground="orange"
            )

        # ----------------------------------------------------
        # SAVE GLOBAL DATA
        # ----------------------------------------------------

        stock_data = data

        actual_prices = y_test.values

        predicted_prices = predictions

        status_label.config(
            text=f"{symbol} data loaded successfully!"
        )

        # Enable buttons
        graph_button.config(
            state="normal"
        )

        prediction_graph_button.config(
            state="normal"
        )

        forecast_button.config(
            state="normal"
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            f"Unable to process stock data.\n\n{error}"
        )

        status_label.config(
            text="Error loading stock data."
        )


# ============================================================
# CREATE GRAPH WINDOW
# ============================================================

def create_graph_window(
    title,
    figure
):

    graph_window = tk.Toplevel(window)

    graph_window.title(title)

    graph_window.geometry(
        "1000x650"
    )

    # Create canvas
    canvas = FigureCanvasTkAgg(
        figure,
        master=graph_window
    )

    canvas.draw()

    canvas.get_tk_widget().pack(
        fill=tk.BOTH,
        expand=True
    )

    # Close figure when window closes
    def close_graph():

        plt.close(figure)

        graph_window.destroy()

    graph_window.protocol(
        "WM_DELETE_WINDOW",
        close_graph
    )


# ============================================================
# SHOW HISTORICAL PRICE GRAPH
# ============================================================

def show_graph():

    if stock_data is None:

        messagebox.showwarning(
            "Warning",
            "Please load stock data first."
        )

        return

    symbol = stock_var.get()

    figure = plt.Figure(
        figsize=(10, 6),
        dpi=100
    )

    axis = figure.add_subplot(111)

    axis.plot(
        stock_data["Date"],
        stock_data["Close"],
        linewidth=2
    )

    axis.set_title(
        f"{symbol} Historical Stock Price"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Stock Price"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    create_graph_window(
        "Historical Stock Price",
        figure
    )


# ============================================================
# SHOW ACTUAL VS PREDICTED GRAPH
# ============================================================

def show_prediction_graph():

    if actual_prices is None:

        messagebox.showwarning(
            "Warning",
            "Please load stock data first."
        )

        return

    figure = plt.Figure(
        figsize=(10, 6),
        dpi=100
    )

    axis = figure.add_subplot(111)

    axis.plot(
        actual_prices,
        label="Actual Price",
        linewidth=2
    )

    axis.plot(
        predicted_prices,
        label="Predicted Price",
        linewidth=2
    )

    axis.set_title(
        "Actual vs Predicted Stock Price"
    )

    axis.set_xlabel(
        "Test Data"
    )

    axis.set_ylabel(
        "Stock Price"
    )

    axis.legend()

    axis.grid(
        True
    )

    figure.tight_layout()

    create_graph_window(
        "Actual vs Predicted",
        figure
    )


# ============================================================
# FUTURE FORECASTING
# ============================================================

def forecast_future():

    global future_prices

    if stock_data is None:

        messagebox.showwarning(
            "Warning",
            "Please load stock data first."
        )

        return

    try:

        days_text = forecast_days_var.get()

        forecast_days = int(
            days_text
        )

        if forecast_days < 1 or forecast_days > 30:

            messagebox.showerror(
                "Error",
                "Please select between 1 and 30 days."
            )

            return

        status_label.config(
            text="Generating future forecast..."
        )

        window.update_idletasks()

        # ----------------------------------------------------
        # Future forecasting uses only features available
        # without knowing future Open/High/Low/Volume.
        # ----------------------------------------------------

        data = stock_data.copy()

        data["Previous_Close"] = (
            data["Close"].shift(1)
        )

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

        data.dropna(
            inplace=True
        )

        future_features = [
            "Previous_Close",
            "MA_5",
            "MA_20"
        ]

        X_future = data[
            future_features
        ]

        y_future = data["Close"]

        # Train future forecasting model
        future_model = LinearRegression()

        future_model.fit(
            X_future,
            y_future
        )

        # Historical prices
        history = list(
            data["Close"].astype(float)
        )

        future_prices = []

        # ----------------------------------------------------
        # RECURSIVE FORECAST
        # ----------------------------------------------------

        for day in range(
            forecast_days
        ):

            previous_close = history[-1]

            ma_5 = np.mean(
                history[-5:]
            )

            ma_20 = np.mean(
                history[-20:]
            )

            input_data = pd.DataFrame({

                "Previous_Close": [
                    previous_close
                ],

                "MA_5": [
                    ma_5
                ],

                "MA_20": [
                    ma_20
                ]

            })

            prediction = future_model.predict(
                input_data
            )[0]

            future_prices.append(
                float(prediction)
            )

            # Add prediction to history
            history.append(
                float(prediction)
            )

        # ----------------------------------------------------
        # DISPLAY FUTURE FORECAST
        # ----------------------------------------------------

        forecast_text.delete(
            "1.0",
            tk.END
        )

        forecast_text.insert(
            tk.END,
            "FUTURE PRICE FORECAST\n"
        )

        forecast_text.insert(
            tk.END,
            "=====================\n\n"
        )

        for index, price in enumerate(
            future_prices,
            start=1
        ):

            forecast_text.insert(
                tk.END,
                f"Day {index}: {price:.2f}\n"
            )

        # ----------------------------------------------------
        # SAVE FORECAST
        # ----------------------------------------------------

        forecast_data = pd.DataFrame({

            "Day": range(
                1,
                forecast_days + 1
            ),

            "Predicted Price": future_prices

        })

        forecast_data.to_csv(
            "future_forecast.csv",
            index=False
        )

        status_label.config(
            text="Future forecast generated successfully!"
        )

        # Enable forecast graph
        future_graph_button.config(
            state="normal"
        )

    except Exception as error:

        messagebox.showerror(
            "Forecast Error",
            str(error)
        )


# ============================================================
# SHOW FUTURE FORECAST GRAPH
# ============================================================

def show_future_graph():

    if future_prices is None:

        messagebox.showwarning(
            "Warning",
            "Generate a future forecast first."
        )

        return

    figure = plt.Figure(
        figsize=(10, 6),
        dpi=100
    )

    axis = figure.add_subplot(111)

    days = list(
        range(
            1,
            len(future_prices) + 1
        )
    )

    axis.plot(
        days,
        future_prices,
        marker="o",
        linewidth=2
    )

    axis.set_title(
        f"{stock_var.get()} Future Price Forecast"
    )

    axis.set_xlabel(
        "Future Day"
    )

    axis.set_ylabel(
        "Predicted Price"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    create_graph_window(
        "Future Price Forecast",
        figure
    )


# ============================================================
# MAIN WINDOW
# ============================================================

window = tk.Tk()

window.title(
    "Stock Price Forecasting System"
)

window.geometry(
    "1000x850"
)

window.resizable(
    False,
    False
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    window,
    text="STOCK PRICE FORECASTING",
    font=("Arial", 26, "bold")
)

title_label.pack(
    pady=(20, 5)
)


subtitle_label = tk.Label(
    window,
    text="Machine Learning Based Stock Prediction System",
    font=("Arial", 13)
)

subtitle_label.pack(
    pady=(0, 15)
)


# ============================================================
# STOCK SELECTION
# ============================================================

selection_frame = tk.Frame(
    window
)

selection_frame.pack(
    pady=10
)

tk.Label(
    selection_frame,
    text="Select Stock:",
    font=("Arial", 13)
).grid(
    row=0,
    column=0,
    padx=10
)


stock_var = tk.StringVar()

stock_combo = ttk.Combobox(
    selection_frame,
    textvariable=stock_var,
    values=[
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS"
    ],
    state="readonly",
    width=20
)

stock_combo.grid(
    row=0,
    column=1,
    padx=10
)

stock_combo.current(
    0
)


# ============================================================
# MAIN BUTTONS
# ============================================================

button_frame = tk.Frame(
    window
)

button_frame.pack(
    pady=10
)


load_button = tk.Button(
    button_frame,
    text="Load Data & Predict",
    command=load_data_and_predict,
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

load_button.grid(
    row=0,
    column=0,
    padx=5
)


graph_button = tk.Button(
    button_frame,
    text="Show Price Graph",
    command=show_graph,
    font=("Arial", 11),
    width=18,
    state="disabled"
)

graph_button.grid(
    row=0,
    column=1,
    padx=5
)


prediction_graph_button = tk.Button(
    button_frame,
    text="Prediction Graph",
    command=show_prediction_graph,
    font=("Arial", 11),
    width=18,
    state="disabled"
)

prediction_graph_button.grid(
    row=0,
    column=2,
    padx=5
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

results_frame = tk.LabelFrame(
    window,
    text="Prediction Results",
    font=("Arial", 14, "bold"),
    padx=30,
    pady=15
)

results_frame.pack(
    pady=15
)


current_price_label = tk.Label(
    results_frame,
    text="Current Price: ---",
    font=("Arial", 16)
)

current_price_label.pack(
    pady=5
)


predicted_price_label = tk.Label(
    results_frame,
    text="Predicted Price: ---",
    font=("Arial", 16)
)

predicted_price_label.pack(
    pady=5
)


prediction_label = tk.Label(
    results_frame,
    text="Prediction: ---",
    font=("Arial", 18, "bold")
)

prediction_label.pack(
    pady=5
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

metrics_frame = tk.LabelFrame(
    window,
    text="Model Performance",
    font=("Arial", 14, "bold"),
    padx=30,
    pady=12
)

metrics_frame.pack(
    pady=10
)


mae_label = tk.Label(
    metrics_frame,
    text="MAE: ---",
    font=("Arial", 12)
)

mae_label.grid(
    row=0,
    column=0,
    padx=25
)


rmse_label = tk.Label(
    metrics_frame,
    text="RMSE: ---",
    font=("Arial", 12)
)

rmse_label.grid(
    row=0,
    column=1,
    padx=25
)


r2_label = tk.Label(
    metrics_frame,
    text="R² Score: ---",
    font=("Arial", 12)
)

r2_label.grid(
    row=0,
    column=2,
    padx=25
)


# ============================================================
# FUTURE FORECAST SECTION
# ============================================================

forecast_frame = tk.LabelFrame(
    window,
    text="Future Forecasting",
    font=("Arial", 14, "bold"),
    padx=20,
    pady=10
)

forecast_frame.pack(
    pady=10
)


tk.Label(
    forecast_frame,
    text="Forecast Days:",
    font=("Arial", 11)
).grid(
    row=0,
    column=0,
    padx=5
)


forecast_days_var = tk.StringVar(
    value="7"
)


forecast_days_combo = ttk.Combobox(
    forecast_frame,
    textvariable=forecast_days_var,
    values=[
        "1",
        "3",
        "5",
        "7",
        "10",
        "15",
        "30"
    ],
    state="readonly",
    width=8
)

forecast_days_combo.grid(
    row=0,
    column=1,
    padx=5
)


forecast_button = tk.Button(
    forecast_frame,
    text="Forecast Future Price",
    command=forecast_future,
    font=("Arial", 11, "bold"),
    width=22,
    state="disabled"
)

forecast_button.grid(
    row=0,
    column=2,
    padx=5
)


future_graph_button = tk.Button(
    forecast_frame,
    text="Forecast Graph",
    command=show_future_graph,
    font=("Arial", 11),
    width=16,
    state="disabled"
)

future_graph_button.grid(
    row=0,
    column=3,
    padx=5
)


# ============================================================
# FORECAST TEXT BOX
# ============================================================

forecast_text = tk.Text(
    forecast_frame,
    height=6,
    width=65,
    font=("Consolas", 11)
)

forecast_text.grid(
    row=1,
    column=0,
    columnspan=4,
    pady=10
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    window,
    text="Select a stock and click Load Data & Predict",
    font=("Arial", 11)
)

status_label.pack(
    pady=10
)


# ============================================================
# START APPLICATION
# ============================================================

window.mainloop()