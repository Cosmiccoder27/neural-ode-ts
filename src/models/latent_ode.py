# src/models/latent_ode.py
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.odefunc import ODEFunc

class Encoder(nn.Module):
    """Encodes input sequence into latent hidden state"""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # take last hidden state
        return h_n[-1]  # shape: [batch, hidden_size]

class LatentODE(nn.Module):
    """Latent Neural ODE model"""
    def __init__(self, input_size, hidden_size=32):
        super(LatentODE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.odefunc = ODEFunc(hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        """
        h0 = self.encoder(x)  # initial latent state
        seq_len = x.size(1)
        t = torch.linspace(0, seq_len-1, seq_len).to(x.device)
        h_t = odeint(self.odefunc, h0, t)  # [seq_len, batch, hidden_size]
        y = self.output_layer(h_t[-1])     # take last time step
        return y
