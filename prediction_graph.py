import pandas as pd
import matplotlib.pyplot as plt

print("Loading prediction results...")

data = pd.read_csv("prediction_results.csv")

plt.figure(figsize=(12, 6))

plt.plot(
    data["Actual Price"],
    label="Actual Price",
    linewidth=2
)

plt.plot(
    data["Predicted Price"],
    label="Predicted Price",
    linewidth=2
)

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Test Data")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()

print("Graph saved as actual_vs_predicted.png")