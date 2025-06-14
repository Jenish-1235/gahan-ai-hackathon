# vit_backbone.py
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_l_16, ViT_L_16_Weights
import torch.nn.functional as F

class ViTBackbone(nn.Module):
    def __init__(self, pretrained=True, output_dim=768):
        super().__init__()
        # Use ViT-Large for better performance
        weights = ViT_L_16_Weights.DEFAULT if pretrained else None
        vit = vit_l_16(weights=weights)
        
        # Remove the classification head
        self.backbone = vit
        self.backbone.heads = nn.Identity()
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(1024, output_dim),  # ViT-L has 1024 dim
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Spatial attention for ROI focus
        self.spatial_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B*T, 3, H, W)
        features = self.backbone(x)  # (B*T, 1024)
        
        # Enhance features
        enhanced_features = self.feature_enhancer(features)  # (B*T, output_dim)
        
        # Apply spatial attention (helps focus on ROI)
        attention_weights = self.spatial_attention(enhanced_features)  # (B*T, 1)
        attended_features = enhanced_features * attention_weights  # (B*T, output_dim)
        
        return attended_features