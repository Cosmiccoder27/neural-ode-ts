# src/models/odefunc.py
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Defines the derivative function dh/dt = f(h, t)
    Used by Neural ODEs to evolve hidden state over time.
    """
    def __init__(self, hidden_size=32):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t, h):
        """
        Args:
            t: scalar or tensor representing current time (required by odeint)
            h: hidden state at time t, shape [batch, hidden_size]

        Returns:
            dh/dt, same shape as h
        """
        return self.net(h)
