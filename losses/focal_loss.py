import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss[targets != self.ignore_index].mean()
        elif self.reduction == 'sum':
            return focal_loss[targets != self.ignore_index].sum()
        return focal_loss