# src/models/neural_ode.py
import torch
import torch.nn as nn
from torchdiffeq import odeint

# --- ODE function ---
class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t, h):
        return self.net(h)

# --- ODE Block ---
class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x):
        # x shape: [batch, window_size, hidden_size]
        batch_size, seq_len, hidden_size = x.shape
        h = x[:, -1, :]  # take last time step as initial hidden state
        t = torch.linspace(0, seq_len-1, seq_len).to(x.device)
        out = odeint(self.odefunc, h, t, rtol=self.tol, atol=self.tol)
        # out shape: [seq_len, batch, hidden_size] → reshape to [batch, hidden_size]
        return out[-1]  # last time step as prediction

# --- Neural ODE model ---
class NeuralODE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralODE, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.odeblock = ODEBlock(ODEFunc(hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, window_size, input_size]
        h = self.input_layer(x)
        out = self.odeblock(h)
        y = self.output_layer(out)
        return y
