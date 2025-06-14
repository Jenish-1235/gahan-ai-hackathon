# Full model wrapper (e.g., in model.py)
from models.vit_backbone import ViTBackbone
from models.lstm_temporal_encoder import LSTMTemporalEncoder
from models.detr_decoder import DETRDecoder
from models.cutin_head import CutInHead
import torch.nn as nn




class CutInDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTBackbone(pretrained=True)
        self.temporal = LSTMTemporalEncoder()
        self.decoder = DETRDecoder()
        self.cutin = CutInHead()

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.vit(x)  # (B*T, D)
        features = features.view(B, T, -1)
        temporal_features = self.temporal(features)  # (B, D)
        class_logits, bboxes, cutin_scores = self.decoder(temporal_features)
        return class_logits, bboxes, cutin_scores

