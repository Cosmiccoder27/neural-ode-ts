# src/train.py
import torch
import torch.nn as nn
from torchdiffeq import odeint  # Neural ODE solver
from src.data import load_data
from src.models.neural_ode import NeuralODE
import os

# --- Hyperparameters ---
DATA_PATH = "data/sample_large.csv"
FEATURE_COL = "value"
WINDOW_SIZE = 20
TEST_SIZE = 0.2
HIDDEN_SIZE = 32
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Prepare directories ---
os.makedirs("results", exist_ok=True)

# --- Load data ---
X_train, y_train, X_test, y_test, scaler = load_data(
    DATA_PATH, feature_col=FEATURE_COL, window_size=WINDOW_SIZE, test_size=TEST_SIZE
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# --- Define model ---
input_size = X_train.shape[2]  # number of features
model = NeuralODE(input_size=input_size, hidden_size=HIDDEN_SIZE).to(DEVICE)

# --- Loss and optimizer ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.6f}")

# --- Save model ---
torch.save(model.state_dict(), "results/neural_ode_model.pt")
print("Training finished and model saved to results/neural_ode_model.pt")
