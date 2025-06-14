import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Ensure consistent absolute path for Colab and CLI
SRC_DIR = '/content/src'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.cutin_sequence_dataset import CutInSequenceDataset
from models.model import CutInDetectionModel
from utils.hungarian_matcher import HungarianMatcher
from utils.losses import CombinedLoss

def collate_fn(batch):
    """Enhanced collate function for proper batching"""
    images, annotations = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Process annotations into proper format
    targets = []
    class_mapping = {'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3}
    
    for batch_idx, frame_annotations in enumerate(annotations):
        # Collect all objects from all frames in the sequence
        all_objects = []
        for frame_ann in frame_annotations:
            all_objects.extend(frame_ann)
        
        if len(all_objects) == 0:
            # Empty target
            targets.append({
                'labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.zeros((0, 4)),
                'cutin': torch.tensor([], dtype=torch.float)
            })
            continue
        
        # Extract labels, boxes, and cutin flags
        labels = []
        boxes = []
        cutin_flags = []
        
        for obj in all_objects:
            # Map class name to index
            class_idx = class_mapping.get(obj['label'], 0)  # 0 for background/unknown
            labels.append(class_idx)
            
            # Use normalized bounding boxes
            boxes.append(obj['bbox_norm'])
            
            # Cut-in flag
            cutin_flags.append(1.0 if obj['cutting'] else 0.0)
        
        targets.append({
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float),
            'cutin': torch.tensor(cutin_flags, dtype=torch.float)
        })
    
    return images, targets

def calculate_metrics(outputs, targets, indices, threshold=0.5):
    """Calculate F1 score and other metrics"""
    all_pred_cutin = []
    all_true_cutin = []
    
    pred_cutin = outputs['pred_cutin']
    
    for batch_idx, (pred_idx, target_idx) in enumerate(indices):
        if len(pred_idx) > 0:
            # Get predictions for matched objects
            batch_pred_cutin = torch.sigmoid(pred_cutin[batch_idx, pred_idx])
            batch_true_cutin = targets[batch_idx]['cutin'][target_idx]
            
            # Convert to binary predictions
            binary_pred = (batch_pred_cutin > threshold).float()
            
            all_pred_cutin.extend(binary_pred.cpu().numpy())
            all_true_cutin.extend(batch_true_cutin.cpu().numpy())
    
    if len(all_pred_cutin) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_cutin, all_pred_cutin, average='binary', zero_division=0
    )
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, dataloader, optimizer, criterion, matcher, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_losses = {'total_loss': [], 'class_loss': [], 'bbox_loss': [], 'giou_loss': [], 'cutin_loss': []}
    all_metrics = {'f1': [], 'precision': [], 'recall': []}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)  # (B, T, C, H, W)
        
        # Move targets to device
        for target in targets:
            target['labels'] = target['labels'].to(device)
            target['boxes'] = target['boxes'].to(device)
            target['cutin'] = target['cutin'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Hungarian matching
        indices = matcher.forward(outputs, targets)
        
        # Calculate loss
        losses = criterion(outputs, targets, indices)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += losses['total_loss'].item()
        for key in all_losses:
            if key in losses:
                all_losses[key].append(losses[key].item())
        
        # Calculate F1 score
        metrics = calculate_metrics(outputs, targets, indices)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{losses['total_loss'].item():.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'CutIn': f"{losses['cutin_loss'].item():.4f}"
        })
        
        # Log to wandb every 50 steps
        if batch_idx % 50 == 0:
            wandb.log({
                'train/step_loss': losses['total_loss'].item(),
                'train/step_f1': metrics['f1'],
                'train/step_cutin_loss': losses['cutin_loss'].item(),
                'train/step': epoch * len(dataloader) + batch_idx
            })
    
    # Calculate epoch averages
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_losses = {key: np.mean(values) for key, values in all_losses.items()}
    
    return avg_loss, avg_metrics, avg_losses

def validate(model, dataloader, criterion, matcher, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    all_metrics = {'f1': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            
            # Move targets to device
            for target in targets:
                target['labels'] = target['labels'].to(device)
                target['boxes'] = target['boxes'].to(device)
                target['cutin'] = target['cutin'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Hungarian matching
            indices = matcher.forward(outputs, targets)
            
            # Calculate loss
            losses = criterion(outputs, targets, indices)
            total_loss += losses['total_loss'].item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets, indices)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_loss, avg_metrics

def main():
    # Initialize wandb
    wandb.init(
        project="cutin-detection-hackathon",
        config={
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 50,
            "model": "ViT-L + GRU + DETR",
            "sequence_length": 5,
            "roi_filter": True,
            "class_balance": True
        }
    )
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced transforms
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets with all optimizations
    train_dataset = CutInSequenceDataset(
        root_dir="/content/distribution/Train",
        transform=transform,
        roi_filter=True,
        balance_classes=True,
        augment=True
    )
    
    val_dataset = CutInSequenceDataset(
        root_dir="/content/distribution/Val",
        transform=transform,
        roi_filter=True,
        balance_classes=False,  # Don't balance validation set
        augment=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Model
    model = CutInDetectionModel(num_classes=4, num_queries=100, hidden_dim=512)
    model.to(device)
    
    # Hungarian matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_cutin=10.0  # High cost for cut-in mismatches
    )
    
    # Advanced loss function
    criterion = CombinedLoss(
        class_weight=1.0,
        bbox_weight=5.0,
        giou_weight=2.0,
        cutin_weight=10.0,  # High weight for cut-in loss
        cutin_pos_weight=10.0  # 10x penalty for false negatives
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50, 
        eta_min=1e-6
    )
    
    # Training loop
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}/50")
        
        # Train
        train_loss, train_metrics, train_losses = train_epoch(
            model, train_loader, optimizer, criterion, matcher, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, matcher, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/f1': train_metrics['f1'],
            'train/precision': train_metrics['precision'],
            'train/recall': train_metrics['recall'],
            'val/loss': val_loss,
            'val/f1': val_metrics['f1'],
            'val/precision': val_metrics['precision'],
            'val/recall': val_metrics['recall'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, 'best_model.pth')
            patience_counter = 0
            print(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
    
    print(f"Training completed! Best F1: {best_f1:.4f}")
    wandb.finish()

if __name__ == '__main__':
    main()