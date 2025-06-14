import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GIoULoss(nn.Module):
    """Generalized IoU Loss for better bounding box regression"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] in format (x1, y1, x2, y2)
            target_boxes: [N, 4] in format (x1, y1, x2, y2)
        """
        # Calculate intersection
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # Calculate union
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area1 + area2 - inter
        
        # Calculate IoU
        iou = inter / union
        
        # Calculate enclosing box
        enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        # Calculate GIoU
        giou = iou - (enclose_area - union) / enclose_area
        
        return 1 - giou.mean()

class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss with 10x penalty for false negatives"""
    
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs if not already applied
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Calculate BCE loss manually with custom weighting
        eps = 1e-8
        loss = -(self.pos_weight * targets * torch.log(inputs + eps) + 
                (1 - targets) * torch.log(1 - inputs + eps))
        
        return loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss function for the cut-in detection model"""
    
    def __init__(self, 
                 class_weight=1.0, 
                 bbox_weight=5.0, 
                 giou_weight=2.0, 
                 cutin_weight=10.0,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 cutin_pos_weight=10.0):
        super().__init__()
        
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.cutin_weight = cutin_weight
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.l1_loss = nn.L1Loss()
        self.giou_loss = GIoULoss()
        self.cutin_loss = WeightedBCELoss(pos_weight=cutin_pos_weight)
    
    def forward(self, outputs, targets, indices):
        """
        Args:
            outputs: dict with 'pred_logits', 'pred_boxes', 'pred_cutin'
            targets: list of target dicts
            indices: list of (pred_idx, target_idx) tuples from Hungarian matching
        """
        # Get matched predictions and targets
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_cutin = outputs['pred_cutin']
        
        # Prepare matched targets
        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]
        
        # Initialize target tensors
        target_classes = torch.full((batch_size, num_queries), 
                                   fill_value=0, dtype=torch.long, 
                                   device=pred_logits.device)  # 0 for background
        target_boxes = torch.zeros((batch_size, num_queries, 4), 
                                  device=pred_boxes.device)
        target_cutin = torch.zeros((batch_size, num_queries), 
                                  device=pred_cutin.device)
        
        # Fill matched targets
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[batch_idx, pred_idx] = targets[batch_idx]['labels'][target_idx]
                target_boxes[batch_idx, pred_idx] = targets[batch_idx]['boxes'][target_idx]
                target_cutin[batch_idx, pred_idx] = targets[batch_idx]['cutin'][target_idx].float()
        
        # Calculate losses
        losses = {}
        
        # Classification loss (Focal Loss)
        class_loss = self.focal_loss(
            pred_logits.view(-1, pred_logits.shape[-1]),
            target_classes.view(-1)
        )
        losses['class_loss'] = class_loss * self.class_weight
        
        # Only calculate bbox and cutin losses for matched predictions
        matched_pred_boxes = []
        matched_target_boxes = []
        matched_pred_cutin = []
        matched_target_cutin = []
        
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                matched_pred_boxes.append(pred_boxes[batch_idx, pred_idx])
                matched_target_boxes.append(target_boxes[batch_idx, pred_idx])
                matched_pred_cutin.append(pred_cutin[batch_idx, pred_idx])
                matched_target_cutin.append(target_cutin[batch_idx, pred_idx])
        
        if matched_pred_boxes:
            matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)
            matched_target_boxes = torch.cat(matched_target_boxes, dim=0)
            matched_pred_cutin = torch.cat(matched_pred_cutin, dim=0)
            matched_target_cutin = torch.cat(matched_target_cutin, dim=0)
            
            # Bounding box L1 loss
            bbox_loss = self.l1_loss(matched_pred_boxes, matched_target_boxes)
            losses['bbox_loss'] = bbox_loss * self.bbox_weight
            
            # GIoU loss
            giou_loss = self.giou_loss(
                self.box_cxcywh_to_xyxy(matched_pred_boxes),
                self.box_cxcywh_to_xyxy(matched_target_boxes)
            )
            losses['giou_loss'] = giou_loss * self.giou_weight
            
            # Cut-in loss (Weighted BCE with 10x penalty for false negatives)
            cutin_loss = self.cutin_loss(matched_pred_cutin, matched_target_cutin)
            losses['cutin_loss'] = cutin_loss * self.cutin_weight
        else:
            # No matched predictions
            losses['bbox_loss'] = torch.tensor(0.0, device=pred_logits.device)
            losses['giou_loss'] = torch.tensor(0.0, device=pred_logits.device)
            losses['cutin_loss'] = torch.tensor(0.0, device=pred_logits.device)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def box_cxcywh_to_xyxy(self, x):
        """Convert boxes from center format to corner format"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1) 