import torch
from src.models.neural_ode import NeuralODE
from src.data import load_data
import matplotlib.pyplot as plt

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
input_size = 1       # same as training
hidden_size = 32     # same as training

model = NeuralODE(input_size=input_size, hidden_size=hidden_size).to(DEVICE)
model.load_state_dict(torch.load("results/neural_ode_model.pt", map_location=DEVICE))
model.eval()

print("=== Model Architecture ===")
print(model)

# --- Load data ---
X_train, y_train, X_test, y_test, scaler = load_data("data/sample_large.csv")

# Convert test data to tensor
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# --- Make predictions ---
with torch.no_grad():
    y_pred = model(X_test)

# Inverse scale predictions
y_pred_np = scaler.inverse_transform(y_pred.cpu().numpy())
y_test_np = scaler.inverse_transform(y_test)

# --- Print first 10 predictions ---
print("\n=== First 10 Predictions vs Actual ===")
for i in range(10):
    print(f"Predicted: {y_pred_np[i][0]:.4f}  |  Actual: {y_test_np[i][0]:.4f}")

# --- Plot full comparison ---
plt.figure(figsize=(10, 5))
plt.plot(y_test_np, label="Actual", color='blue')
plt.plot(y_pred_np, label="Predicted", color='red', linestyle='--')
plt.title("Neural ODE Predictions vs Actual")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()
