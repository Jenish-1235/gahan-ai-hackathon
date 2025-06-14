import torch
import torch.nn as nn
from .vit_backbone import ViTBackbone
from .temporal_encoder import TemporalEncoder
from .detr_decoder import DETRDecoder

class CuttingDetector(nn.Module):
    """
    Full model: ViT backbone + GRU temporal encoder + DETR decoder.
    """
    def __init__(
        self,
        vit_model_name="vit_base_patch16_224_in21k",
        vit_output_dim=768,
        temporal_hidden_dim=256,
        temporal_layers=2,
        temporal_dropout=0.1,
        temporal_attention_heads=4,
        use_temporal_attention=True,
        num_queries=100,
        num_classes=4,
        detr_hidden_dim=256,
        detr_heads=8,
        detr_decoder_layers=6
    ):
        super().__init__()
        self.vit = ViTBackbone(model_name=vit_model_name, pretrained=True, output_dim=vit_output_dim)
        self.temporal_encoder = TemporalEncoder(
            feature_dim=vit_output_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            use_attention=use_temporal_attention,
            num_heads=temporal_attention_heads
        )
        self.detr_decoder = DETRDecoder(
            input_dim=temporal_hidden_dim * 2,  # bidirectional GRU
            num_queries=num_queries,
            num_classes=num_classes,
            hidden_dim=detr_hidden_dim,
            nheads=detr_heads,
            num_decoder_layers=detr_decoder_layers
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of image sequences
        Returns:
            dict with keys: pred_logits, pred_boxes, pred_cutting
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        vit_feats = self.vit(x)  # (B*T, N_patches, D)
        N_patches, D = vit_feats.shape[1], vit_feats.shape[2]
        vit_feats = vit_feats.view(B, T, N_patches, D)  # (B, T, N_patches, D)
        temporal_feats = self.temporal_encoder(vit_feats)  # (B, T, hidden_dim*2)
        return self.detr_decoder(temporal_feats)