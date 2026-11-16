import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Model results
models = ["Linear Regression", "Random Forest"]

mae = [1.02, 10.40]
rmse = [1.41, 16.99]
r2 = [0.9967, 0.5267]

# -----------------------------------------
# MAE COMPARISON
# -----------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(models, mae)

plt.title("MAE Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error")

plt.tight_layout()
plt.savefig("mae_comparison.png")
plt.show()

# -----------------------------------------
# RMSE COMPARISON
# -----------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(models, rmse)

plt.title("RMSE Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Root Mean Squared Error")

plt.tight_layout()
plt.savefig("rmse_comparison.png")
plt.show()

# -----------------------------------------
# R2 COMPARISON
# -----------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(models, r2)

plt.title("R² Score Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")

plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.show()

print("Model comparison graphs created successfully!")