import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.hungarian_matcher import HungarianMatcher
from .focal_loss import FocalLoss

class DetectionCriterion(nn.Module):
    """
    Combined loss for DETR-style detection with class imbalance handling.
    """
    def __init__(self, num_classes, matcher, 
                 ce_loss_coef=1.0, bbox_loss_coef=5.0, giou_loss_coef=2.0, cutting_loss_coef=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.ce_loss_coef = ce_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.cutting_loss_coef = cutting_loss_coef
        self.eos_coef = eos_coef
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bbox_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))  # Heavily weight positives

    def forward(self, outputs, targets):
        """
        outputs: dict from model
        targets: list of dicts, one per batch element
        """
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)

        # Classification loss (focal)
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[batch_idx, src_idx] = targets[batch_idx]['labels'][tgt_idx]
        loss_ce = self.focal_loss(src_logits.flatten(0, 1), target_classes.flatten(0, 1))

        # Bounding box loss
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = self.bbox_loss(src_boxes, target_boxes)

        # Cutting loss (weighted BCE)
        src_cutting = outputs['pred_cutting'][idx]
        target_cutting = torch.cat([t['cutting_flags'][i].float() for t, (_, i) in zip(targets, indices)], dim=0)
        loss_cutting = self.bce_loss(src_cutting, target_cutting)

        # Total loss
        loss = (self.ce_loss_coef * loss_ce +
                self.bbox_loss_coef * loss_bbox +
                self.cutting_loss_coef * loss_cutting)
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return (batch_idx, src_idx)