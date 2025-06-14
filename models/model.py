import torch
import torch.nn as nn
from models.vit_backbone import ViTBackbone
from models.gru_temporal_encoder import GRUTemporalEncoder
from models.detr_decoder import DETRDecoder
from utils.roi_cropper import ROICropper




class OptimizedCutInDetectionModel(nn.Module):
    def __init__(self, num_classes=4, num_queries=100):
        super().__init__()
        self.vit = ViTBackbone(pretrained=True)
        self.temporal = GRUTemporalEncoder(input_dim=768, hidden_dim=512)
        self.decoder = DETRDecoder(input_dim=512, num_queries=num_queries, num_classes=num_classes)
        self.roi_cropper = ROICropper()
        
        # Add attention mechanism for better feature fusion
        self.feature_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
    def forward(self, x, annotations=None):
        B, T, C, H, W = x.shape
        
        # Apply ROI cropping if annotations available
        if annotations is not None and self.training:
            cropped_sequences = []
            for seq_idx in range(T):
                seq_images = x[:, seq_idx]  # (B, C, H, W)
                seq_annotations = [ann[seq_idx] if seq_idx < len(ann) else [] 
                                 for ann in annotations]
                cropped = self.roi_cropper.crop_around_objects(seq_images, seq_annotations)
                cropped_sequences.append(cropped)
            x = torch.stack(cropped_sequences, dim=1)  # (B, T, C, H, W)
        
        # Reshape for ViT processing
        x = x.view(B * T, C, H, W)
        features = self.vit(x)  # (B*T, D)
        features = features.view(B, T, -1)
        
        # Temporal encoding with GRU
        temporal_features = self.temporal(features)  # (B, D)
        
        # Apply self-attention for better representation
        attended_features, _ = self.feature_attention(
            temporal_features.unsqueeze(1), 
            temporal_features.unsqueeze(1), 
            temporal_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # DETR decoder
        class_logits, bboxes, cutin_scores = self.decoder(attended_features)
        
        return {
            "pred_logits": class_logits,
            "pred_boxes": bboxes,
            "pred_cutin": cutin_scores
        }

