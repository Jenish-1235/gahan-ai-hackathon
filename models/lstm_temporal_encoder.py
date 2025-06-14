import torch
import torch.nn as nn

class LSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (B, hidden_dim)