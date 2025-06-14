# cutin_head.py
import torch
import torch.nn as nn

class CutInHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.cutin_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, query_features):
        return self.cutin_classifier(query_features).squeeze(-1)  # (B, Q)
