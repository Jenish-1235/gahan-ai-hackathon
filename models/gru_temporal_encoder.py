import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        # Bidirectional GRU for better temporal context
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention mechanism for temporal features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # GRU processing
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim * 2)
        
        # Self-attention over temporal dimension
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)  # (B, T, hidden_dim * 2)
        
        # Combine GRU and attention outputs
        combined = gru_out + attn_out  # Residual connection
        
        # Use the last timestep for final representation
        final_features = combined[:, -1, :]  # (B, hidden_dim * 2)
        
        # Project to desired output dimension
        output = self.output_proj(final_features)  # (B, hidden_dim)
        
        return output 