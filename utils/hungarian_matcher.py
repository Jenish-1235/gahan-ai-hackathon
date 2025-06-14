import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, cost_cutin=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_cutin = cost_cutin
        
    def forward(self, outputs, targets):
        """Performs the matching between predictions and targets"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            
            # Flatten to compute the cost matrices
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            out_cutin = outputs["pred_cutin"].flatten(0, 1)  # [batch_size * num_queries]
            
            indices = []
            for i, target in enumerate(targets):
                if len(target["labels"]) == 0:
                    indices.append((torch.tensor([]), torch.tensor([])))
                    continue
                    
                tgt_ids = target["labels"]
                tgt_bbox = target["boxes"]
                tgt_cutin = target["cutting"]
                
                # Compute the classification cost
                cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]
                
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)
                
                # Compute the GIoU cost between boxes
                cost_giou = -self.generalized_box_iou(
                    out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox
                )
                
                # Compute cutin cost
                cost_cutin = torch.abs(
                    out_cutin[i * num_queries:(i + 1) * num_queries].unsqueeze(1) - tgt_cutin.unsqueeze(0)
                )
                
                # Final cost matrix
                C = (self.cost_bbox * cost_bbox + 
                     self.cost_class * cost_class + 
                     self.cost_giou * cost_giou +
                     self.cost_cutin * cost_cutin)
                
                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
                indices.append((torch.tensor(row_ind), torch.tensor(col_ind)))
                
        return indices
    
    def generalized_box_iou(self, boxes1, boxes2):
        """Compute generalized IoU between two sets of boxes"""
        # Implementation of GIoU
        # Simplified version - you can use torchvision.ops.generalized_box_iou
        from torchvision.ops import generalized_box_iou
        return generalized_box_iou(boxes1, boxes2)