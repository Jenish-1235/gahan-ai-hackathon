# detr_decoder.py
import torch
import torch.nn as nn

class DETRDecoder(nn.Module):
    def __init__(self, input_dim=512, num_queries=100, num_classes=4):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, input_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=8), num_layers=6
        )
        self.class_embed = nn.Linear(input_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 4),
            nn.Sigmoid()  # normalized bbox
        )
        self.cutin_head = nn.Linear(input_dim, 1)

    def forward(self, memory):
      B = memory.size(0)
      Q = self.query_embed.num_embeddings
      tgt = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, D]
      memory = memory.unsqueeze(0)  # [1, B, D]
      
      decoded = self.decoder(tgt, memory)  # [Q, B, D]
      decoded = decoded.transpose(0, 1)    # [B, Q, D]
      
      logits = self.class_embed(decoded)       # [B, Q, num_classes]
      bboxes = self.bbox_embed(decoded).sigmoid()  # [B, Q, 4]
      cutin_scores = self.cutin_head(decoded).squeeze(-1).sigmoid()  # [B, Q]

      return logits, bboxes, cutin_scores

