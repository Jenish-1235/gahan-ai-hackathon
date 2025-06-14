import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_cutting_metrics(outputs, targets, threshold=0.5):
    """
    Computes precision, recall, and F1 for lane cutting detection.
    Args:
        outputs: dict from model, contains 'pred_cutting' (B, num_queries)
        targets: list of dicts, each with 'cutting_flags'
        threshold: threshold for positive prediction
    Returns:
        precision, recall, f1
    """
    pred_cutting = torch.sigmoid(outputs['pred_cutting']).cpu().numpy() > threshold
    true_cutting = []
    pred_cutting_flat = []

    for b, target in enumerate(targets):
        n = len(target['cutting_flags'])
        if n == 0:
            continue
        true_cutting.extend(target['cutting_flags'].cpu().numpy())
        pred_cutting_flat.extend(pred_cutting[b][:n])

    if not true_cutting:
        return 0.0, 0.0, 0.0

    precision = precision_score(true_cutting, pred_cutting_flat, zero_division=0)
    recall = recall_score(true_cutting, pred_cutting_flat, zero_division=0)
    f1 = f1_score(true_cutting, pred_cutting_flat, zero_division=0)
    return precision, recall, f1