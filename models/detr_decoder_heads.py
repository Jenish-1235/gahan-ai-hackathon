import torch
import torch.nn as nn

class DETRDecoder(nn.Module):
    """
    DETR-style transformer decoder for object queries.
    Predicts class, bounding box, and lane cutting flag for each query.
    """
    def __init__(self, 
                 input_dim=512, 
                 num_queries=100, 
                 num_classes=4, 
                 hidden_dim=256, 
                 nheads=8, 
                 num_decoder_layers=6):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # Normalized box coordinates
        )
        self.cutting_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, temporal_features):
        """
        Args:
            temporal_features: (B, T, D) - output from TemporalEncoder
        Returns:
            dict with:
                'pred_logits': (B, num_queries, num_classes)
                'pred_boxes': (B, num_queries, 4)
                'pred_cutting': (B, num_queries)
        """
        B, T, D = temporal_features.shape
        # Pool temporal features (e.g., last frame or mean)
        pooled = temporal_features.mean(dim=1)  # (B, D)
        src = self.input_proj(pooled).unsqueeze(1)  # (B, 1, hidden_dim)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, hidden_dim)

        tgt = torch.zeros_like(query_embed)  # (B, num_queries, hidden_dim)
        hs = self.decoder(tgt, src.repeat(1, query_embed.size(1), 1))  # (B, num_queries, hidden_dim)

        pred_logits = self.class_embed(hs)  # (B, num_queries, num_classes)
        pred_boxes = self.bbox_embed(hs)    # (B, num_queries, 4)
        pred_cutting = self.cutting_embed(hs).squeeze(-1)  # (B, num_queries)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_cutting": pred_cutting
        }