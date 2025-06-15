import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DETR-style assignment.
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_cutting=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_cutting = cost_cutting

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (bs, num_queries, num_classes)
        out_bbox = outputs["pred_boxes"]               # (bs, num_queries, 4)
        out_cutting = outputs["pred_cutting"]          # (bs, num_queries)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']
            tgt_cutting = targets[b]['cutting_flags'].float()
            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Classification cost: negative log-likelihood
            cost_class = -out_prob[b][:, tgt_ids]

            # Bbox cost: L1 distance
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

            # Cutting cost: absolute difference
            cost_cutting = torch.abs(out_cutting[b].unsqueeze(1) - tgt_cutting.unsqueeze(0))

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_cutting * cost_cutting
            C = C.cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices