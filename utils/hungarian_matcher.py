import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianMatcher:
    """Hungarian matcher for DETR-style object detection"""
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_cutin=10.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_cutin = cost_cutin
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching
        
        Args:
            outputs: dict with keys:
                - "pred_logits": [B, num_queries, num_classes]
                - "pred_boxes": [B, num_queries, 4]
                - "pred_cutin": [B, num_queries]
            targets: list of dicts, each with keys:
                - "labels": [num_objects]
                - "boxes": [num_objects, 4]
                - "cutin": [num_objects]
        
        Returns:
            List of tuples (pred_indices, target_indices) for each batch
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]
        out_cutin = outputs["pred_cutin"].flatten(0, 1).sigmoid()  # [B*num_queries]
        
        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_cutin = torch.cat([v["cutin"] for v in targets])
        
        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the GIoU cost between boxes
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Compute the cut-in cost
        cost_cutin = torch.abs(out_cutin.unsqueeze(1) - tgt_cutin.unsqueeze(0))
        
        # Final cost matrix
        C = (self.cost_bbox * cost_bbox + 
             self.cost_class * cost_class + 
             self.cost_giou * cost_giou +
             self.cost_cutin * cost_cutin)
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    def box_cxcywh_to_xyxy(self, x):
        """Convert boxes from center format to corner format"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    def generalized_box_iou(self, boxes1, boxes2):
        """Compute generalized IoU between two sets of boxes"""
        # Ensure boxes are in the correct format
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        iou, union = self.box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]
        
        return iou - (area - union) / area
    
    def box_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        return iou, union
    
    def box_area(self, boxes):
        """Compute area of boxes"""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) 