import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    GRU-based temporal encoder with optional temporal self-attention.
    Processes a sequence of per-frame ViT features.
    """
    def __init__(self, feature_dim=768, hidden_dim=256, num_layers=2, dropout=0.1, use_attention=True, num_heads=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim * 2  # bidirectional

        if use_attention:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )
        else:
            self.temporal_attn = None

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, N_patches, D) or (B, T, D)
               - If (B, T, N_patches, D): flatten to (B*T, N_patches, D), mean-pool over patches.
        Returns:
            Tensor of shape (B, T, hidden_dim*2) after GRU (and attention if enabled)
        """
        if x.dim() == 4:
            # Mean-pool over patches: (B, T, N_patches, D) -> (B, T, D)
            x = x.mean(dim=2)
        # x: (B, T, D)
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim*2)
        if self.use_attention:
            # Self-attention over time (T)
            attn_out, _ = self.temporal_attn(gru_out, gru_out, gru_out)
            return attn_out  # (B, T, hidden_dim*2)
        else:
            return gru_out  # (B, T, hidden_dim*2)