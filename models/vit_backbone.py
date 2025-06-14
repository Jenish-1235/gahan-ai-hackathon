# vit_backbone.py
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class ViTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = vit_b_16(weights=weights)
        self.backbone = vit
        self.backbone.heads = nn.Identity()  # remove classifier head

    def forward(self, x):
        # x: (B*T, 3, H, W)
        return self.backbone(x)  # (B*T, D)