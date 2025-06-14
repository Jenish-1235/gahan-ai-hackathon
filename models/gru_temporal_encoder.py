import torch
import torch.nn as nn

class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional
        
    def forward(self, x):
        # x: (B, T, D)
        out, hidden = self.gru(x)
        # Use last hidden state from both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_dim*2)
        return self.projection(final_hidden)  # (B, hidden_dim)