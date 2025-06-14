# Full model wrapper (e.g., in model.py)
from models.vit_backbone import ViTBackbone
from models.gru_temporal_encoder import GRUTemporalEncoder
from models.detr_decoder import DETRDecoder
import torch.nn as nn
import torch

class CutInDetectionModel(nn.Module):
    def __init__(self, num_classes=4, num_queries=100, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Enhanced components
        self.vit = ViTBackbone(pretrained=True, output_dim=768)
        self.temporal = GRUTemporalEncoder(input_dim=768, hidden_dim=hidden_dim)
        self.decoder = DETRDecoder(input_dim=hidden_dim, num_queries=num_queries, num_classes=num_classes)
        
        # Class mapping
        self.class_names = ['Background', 'Car', 'MotorBike', 'EgoVehicle']

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of image sequences
        
        Returns:
            dict with 'pred_logits', 'pred_boxes', 'pred_cutin'
        """
        B, T, C, H, W = x.shape
        
        # Reshape for ViT processing
        x = x.view(B * T, C, H, W)
        
        # Extract visual features
        features = self.vit(x)  # (B*T, 768)
        
        # Reshape back to sequence format
        features = features.view(B, T, -1)  # (B, T, 768)
        
        # Temporal encoding with GRU
        temporal_features = self.temporal(features)  # (B, hidden_dim)
        
        # DETR-style decoding
        class_logits, bboxes, cutin_scores = self.decoder(temporal_features)
        
        return {
            'pred_logits': class_logits,    # (B, num_queries, num_classes)
            'pred_boxes': bboxes,           # (B, num_queries, 4)
            'pred_cutin': cutin_scores      # (B, num_queries)
        }
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.class_names[class_idx] if 0 <= class_idx < len(self.class_names) else 'Unknown'

