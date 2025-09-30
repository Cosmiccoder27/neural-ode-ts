# src/models/cde_model.py
import torch
import torch.nn as nn
from torchcde import NeuralCDE  # pip install torchcde

class CDEFunc(nn.Module):
    """Defines derivative function for Neural CDE"""
    def __init__(self, hidden_size, input_channels):
        super(CDEFunc, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * input_channels)
        )
        self.hidden_size = hidden_size
        self.input_channels = input_channels

    def forward(self, t, h):
        # h shape: [batch, hidden_size]
        out = self.func(h)
        out = out.view(h.size(0), self.hidden_size, self.input_channels)
        return out

class NeuralCDEModel(nn.Module):
    def __init__(self, input_channels, hidden_size=32):
        super(NeuralCDEModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.func = CDEFunc(hidden_size, input_channels)
        self.initial_linear = nn.Linear(input_channels, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, X):
        """
        X: [batch, seq_len, input_channels]
        """
        X = torchcde.LinearInterpolation(X).evaluate()  # interpolate if needed
        h0 = self.initial_linear(X[:,0,:])
        pred = NeuralCDE(X=X, func=self.func, hidden_size=self.hidden_size, h0=h0).evaluate(X[:,-1,:])
        y = self.readout(pred)
        return y
