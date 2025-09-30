# src/eval.py
import torch
import matplotlib.pyplot as plt
from src.data import load_data
from src.models.neural_ode import NeuralODE
import os
import numpy as np

# --- Hyperparameters ---
DATA_PATH = "data/sample.csv"
FEATURE_COL = "value"
WINDOW_SIZE = 20
TEST_SIZE = 0.2
HIDDEN_SIZE = 32
MODEL_PATH = "results/neural_ode_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data ---
X_train, y_train, X_test, y_test, scaler = load_data(
    DATA_PATH, feature_col=FEATURE_COL, window_size=WINDOW_SIZE, test_size=TEST_SIZE
)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# --- Load model ---
input_size = X_test.shape[2]
model = NeuralODE(input_size=input_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Make predictions ---
with torch.no_grad():
    y_pred = model(X_test)

# Convert back to original scale
y_pred_np = scaler.inverse_transform(y_pred.cpu().numpy())
y_test_np = scaler.inverse_transform(y_test.cpu().numpy())

# --- Compute metrics ---
mse = np.mean((y_pred_np - y_test_np) ** 2)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")

# --- Plot predictions vs actual ---
plt.figure(figsize=(12,6))
plt.plot(y_test_np, label="Actual")
plt.plot(y_pred_np, label="Predicted")
plt.title("Neural ODE Time-Series Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
os.makedirs("results", exist_ok=True)
plt.savefig("results/prediction_plot.png")
plt.show()
print("Plot saved to results/prediction_plot.png")
