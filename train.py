import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from datasets.cutin_sequence_dataset import CutInSequenceDataset
from models.optimized_model import OptimizedCutInDetectionModel
from utils.hungarian_matcher import HungarianMatcher
from utils.focal_loss import FocalLoss, WeightedBCELoss

class OptimizedLoss(nn.Module):
    """Loss module for optimized cut-in detection."""

    def __init__(self, matcher, num_classes=4, cutin_weight=10.0):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.cutin_weight = cutin_weight
        
        # Class imbalance handling
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.bbox_loss = nn.L1Loss()
        self.cutin_loss = WeightedBCELoss(pos_weight=cutin_weight)
        
    def forward(self, outputs, targets):
        """Compute the loss given model outputs and targets."""
        indices = self.matcher(outputs, targets)
        
        # Compute losses
        loss_dict = {}
        
        # Classification loss with focal loss
        src_logits = outputs['pred_logits']
        target_classes = self._get_target_classes(targets, indices, src_logits.shape[0])
        loss_dict['loss_ce'] = self.focal_loss(src_logits.transpose(1, 2), target_classes)
        
        # Bbox loss
        src_boxes = outputs['pred_boxes']
        target_boxes = self._get_target_boxes(targets, indices)
        if target_boxes.numel() > 0:
            loss_dict['loss_bbox'] = self.bbox_loss(src_boxes, target_boxes)
        else:
            loss_dict['loss_bbox'] = torch.tensor(0.0, device=src_boxes.device)
        
        # Cut-in loss with heavy weighting for true cases
        src_cutin = outputs['pred_cutin']
        target_cutin = self._get_target_cutin(targets, indices, src_cutin.shape)
        loss_dict['loss_cutin'] = self.cutin_loss(src_cutin, target_cutin)
        
        # Total loss
        total_loss = (loss_dict['loss_ce'] + 
                     5 * loss_dict['loss_bbox'] + 
                     self.cutin_weight * loss_dict['loss_cutin'])
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def _get_target_classes(self, targets, indices, batch_size):
        target_classes = torch.full((batch_size, 100), self.num_classes, dtype=torch.int64)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_classes[i, src_idx] = targets[i]['labels'][tgt_idx]
        return target_classes
    
    def _get_target_boxes(self, targets, indices):
        # Implementation for target box extraction
        target_boxes = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_boxes.append(targets[i]['boxes'][tgt_idx])
        if target_boxes:
            return torch.cat(target_boxes)
        return torch.empty(0, 4)
    
    def _get_target_cutin(self, targets, indices, shape):
        target_cutin = torch.zeros(shape, dtype=torch.float32)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_cutin[i, src_idx] = targets[i]['cutting'][tgt_idx].float()
        return target_cutin

def convert_annotations_to_targets(annotations_batch):
    """Convert dataset annotations to model target format"""
    targets = []
    
    for sequence_annotations in annotations_batch:
        # Use last frame annotations for sequence-level prediction
        last_frame_ann = sequence_annotations[-1] if sequence_annotations else []
        
        if not last_frame_ann:
            targets.append({
                'labels': torch.empty(0, dtype=torch.long),
                'boxes': torch.empty(0, 4),
                'cutting': torch.empty(0, dtype=torch.bool)
            })
            continue
        
        labels = []
        boxes = []
        cutting = []
        
        # Map object names to class indices
        class_map = {'EgoVehicle': 0, 'Bicycle': 1, 'MotorBike': 2, 'Car': 3}
        
        for obj in last_frame_ann:
            if obj['label'] in class_map:
                labels.append(class_map[obj['label']])
                # Normalize bounding boxes
                bbox = obj['bbox']
                normalized_bbox = [
                    bbox[0] / 1920,  # xmin
                    bbox[1] / 1080,  # ymin  
                    bbox[2] / 1920,  # xmax
                    bbox[3] / 1080   # ymax
                ]
                boxes.append(normalized_bbox)
                cutting.append(obj['cutting'])
        
        targets.append({
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'cutting': torch.tensor(cutting, dtype=torch.bool)
        })
    
    return targets

def compute_metrics(outputs, targets, threshold=0.5):
    """Compute precision, recall, F1 for cut-in detection"""
    pred_cutin = (outputs['pred_cutin'] > threshold).cpu().numpy()
    true_cutin = []
    
    for target in targets:
        if len(target['cutting']) > 0:
            true_cutin.extend(target['cutting'].cpu().numpy())
        else:
            true_cutin.extend([False] * pred_cutin.shape[1])
    
    true_cutin = np.array(true_cutin[:pred_cutin.size])
    pred_cutin = pred_cutin.flatten()[:len(true_cutin)]
    
    precision = precision_score(true_cutin, pred_cutin, zero_division=0)
    recall = recall_score(true_cutin, pred_cutin, zero_division=0)
    f1 = f1_score(true_cutin, pred_cutin, zero_division=0)
    
    return precision, recall, f1

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for i, (images, annotations) in enumerate(dataloader):
        images = images.to(device)
        targets = convert_annotations_to_targets(annotations)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        outputs = model(images, annotations)
        loss_dict = criterion(outputs, targets)
        
        loss = loss_dict['total_loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        precision, recall, f1 = compute_metrics(outputs, targets)
        
        total_loss += loss.item()
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  CE: {loss_dict['loss_ce'].item():.4f}")
            print(f"  BBox: {loss_dict['loss_bbox'].item():.4f}")
            print(f"  CutIn: {loss_dict['loss_cutin'].item():.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        'loss': total_loss / len(dataloader),
        'precision': total_precision / len(dataloader),
        'recall': total_recall / len(dataloader),
        'f1': total_f1 / len(dataloader)
    }

def main():
    # Dataset and DataLoader
    root_dir = "/content/distribution/Train"
    transform = Compose([Resize((224, 224)), ToTensor()])
    dataset = CutInSequenceDataset(root_dir=root_dir, transform=transform, sequence_length=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedCutInDetectionModel(num_classes=4).to(device)
    
    matcher = HungarianMatcher()
    criterion = OptimizedLoss(matcher, num_classes=4, cutin_weight=15.0)
    
    # Optimizer with weight decay and learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_f1 = 0
    for epoch in range(20):
        metrics = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {metrics['loss']:.4f}")
        print(f"  Average Precision: {metrics['precision']:.4f}")
        print(f"  Average Recall: {metrics['recall']:.4f}")
        print(f"  Average F1: {metrics['f1']:.4f}")
        
        scheduler.step(metrics['loss'])
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  New best F1 score: {best_f1:.4f} - Model saved!")

if __name__ == '__main__':
    main()