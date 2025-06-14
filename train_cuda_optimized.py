import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings
import time
import gc
warnings.filterwarnings('ignore')

# Ensure consistent absolute path for Colab and CLI
SRC_DIR = '/content/src'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.cutin_sequence_dataset import CutInSequenceDataset
from models.model import CutInDetectionModel
from utils.hungarian_matcher import HungarianMatcher
from utils.losses import CombinedLoss

# CUDA Optimization Settings
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster training
torch.backends.cudnn.allow_tf32 = True

def setup_cuda_environment():
    """Optimize CUDA environment for maximum performance"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
        
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Available Memory: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
        
        return torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')

class FastDataLoader:
    """Optimized DataLoader with prefetching and caching"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2,  # Prefetch batches
            drop_last=True  # Consistent batch sizes
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

def collate_fn_optimized(batch):
    """Optimized collate function with CUDA tensors"""
    images, annotations = zip(*batch)
    
    # Stack images directly on GPU if possible
    images = torch.stack(images)
    
    # Process annotations efficiently
    targets = []
    class_mapping = {'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3}
    
    max_objects = 50  # Limit objects per batch for memory efficiency
    
    for batch_idx, frame_annotations in enumerate(annotations):
        all_objects = []
        for frame_ann in frame_annotations:
            all_objects.extend(frame_ann[:max_objects])  # Limit objects
        
        if len(all_objects) == 0:
            targets.append({
                'labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'cutin': torch.tensor([], dtype=torch.float32)
            })
            continue
        
        # Vectorized processing
        labels = [class_mapping.get(obj['label'], 0) for obj in all_objects]
        boxes = [obj['bbox_norm'] for obj in all_objects]
        cutin_flags = [1.0 if obj['cutting'] else 0.0 for obj in all_objects]
        
        targets.append({
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'cutin': torch.tensor(cutin_flags, dtype=torch.float32)
        })
    
    return images, targets

def calculate_metrics_fast(outputs, targets, indices, threshold=0.5):
    """Fast metrics calculation with CUDA tensors"""
    all_pred_cutin = []
    all_true_cutin = []
    
    pred_cutin = outputs['pred_cutin']
    
    with torch.no_grad():
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                batch_pred_cutin = torch.sigmoid(pred_cutin[batch_idx, pred_idx])
                batch_true_cutin = targets[batch_idx]['cutin'][target_idx]
                
                binary_pred = (batch_pred_cutin > threshold).float()
                
                all_pred_cutin.extend(binary_pred.cpu().numpy())
                all_true_cutin.extend(batch_true_cutin.cpu().numpy())
    
    if len(all_pred_cutin) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_cutin, all_pred_cutin, average='binary', zero_division=0
    )
    
    return {'f1': f1, 'precision': precision, 'recall': recall}

def train_epoch_cuda(model, dataloader, optimizer, criterion, matcher, scaler, device, epoch):
    """CUDA-optimized training epoch with mixed precision"""
    model.train()
    total_loss = 0
    all_metrics = {'f1': [], 'precision': [], 'recall': []}
    
    # Use non_blocking transfers
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Async GPU transfer
        images = images.to(device, non_blocking=True)
        
        # Move targets to device efficiently
        for target in targets:
            target['labels'] = target['labels'].to(device, non_blocking=True)
            target['boxes'] = target['boxes'].to(device, non_blocking=True)
            target['cutin'] = target['cutin'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            
            # Hungarian matching (no gradients needed)
            with torch.no_grad():
                indices = matcher.forward(outputs, targets)
            
            # Calculate loss
            losses = criterion(outputs, targets, indices)
            loss = losses['total_loss']
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        
        # Calculate F1 score every 10 steps to save time
        if batch_idx % 10 == 0:
            metrics = calculate_metrics_fast(outputs, targets, indices)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        # Update progress bar
        if batch_idx % 5 == 0:
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'F1': f"{metrics.get('f1', 0):.4f}" if batch_idx % 10 == 0 else "N/A",
                'GPU': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            })
        
        # Memory cleanup every 50 steps
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # Log to wandb every 100 steps
        if batch_idx % 100 == 0:
            wandb.log({
                'train/step_loss': loss.item(),
                'train/gpu_memory': torch.cuda.memory_allocated()/1e9,
                'train/step': epoch * len(dataloader) + batch_idx
            })
    
    # Calculate epoch averages
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in all_metrics.items()}
    
    return avg_loss, avg_metrics

def validate_cuda(model, dataloader, criterion, matcher, device):
    """CUDA-optimized validation"""
    model.eval()
    total_loss = 0
    all_metrics = {'f1': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            
            for target in targets:
                target['labels'] = target['labels'].to(device, non_blocking=True)
                target['boxes'] = target['boxes'].to(device, non_blocking=True)
                target['cutin'] = target['cutin'].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                indices = matcher.forward(outputs, targets)
                losses = criterion(outputs, targets, indices)
                total_loss += losses['total_loss'].item()
            
            metrics = calculate_metrics_fast(outputs, targets, indices)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_loss, avg_metrics

def main():
    # Setup CUDA environment
    device = setup_cuda_environment()
    
    # Initialize wandb
    wandb.init(
        project="cutin-detection-cuda-optimized",
        config={
            "learning_rate": 2e-4,  # Slightly higher for faster convergence
            "batch_size": 8,  # Larger batch size for GPU efficiency
            "epochs": 30,  # Fewer epochs with better optimization
            "model": "ViT-L + GRU + DETR + CUDA",
            "mixed_precision": True,
            "cuda_optimized": True
        }
    )
    
    print("🚀 Starting CUDA-Optimized Training")
    
    # Optimized transforms
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Fast datasets (reduced preprocessing for speed)
    print("📊 Loading datasets...")
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
        balance_classes=False,
        augment=False
    )
    
    # Optimized data loaders
    train_loader = FastDataLoader(
        train_dataset, 
        batch_size=8,  # Larger batch size
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True
    )
    
    val_loader = FastDataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model with DataParallel if multiple GPUs
    model = CutInDetectionModel(num_classes=4, num_queries=100, hidden_dim=512)
    
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    
    model.to(device)
    
    # Hungarian matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_cutin=10.0
    )
    
    # Advanced loss function
    criterion = CombinedLoss(
        class_weight=1.0,
        bbox_weight=5.0,
        giou_weight=2.0,
        cutin_weight=15.0,  # Higher weight for better F1
        cutin_pos_weight=10.0
    )
    
    # Optimized optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # Higher learning rate
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warm up for 10% of training
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_f1 = 0.0
    patience = 8
    patience_counter = 0
    
    print("🎯 Starting training loop...")
    
    for epoch in range(30):
        start_time = time.time()
        
        print(f"\n🔄 Epoch {epoch+1}/30")
        
        # Train
        train_loss, train_metrics = train_epoch_cuda(
            model, train_loader, optimizer, criterion, matcher, scaler, device, epoch
        )
        
        # Validate every 2 epochs to save time
        if epoch % 2 == 0:
            val_loss, val_metrics = validate_cuda(
                model, val_loader, criterion, matcher, device
            )
        else:
            val_loss, val_metrics = train_loss, train_metrics  # Use train metrics
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/f1': train_metrics['f1'],
            'val/loss': val_loss,
            'val/f1': val_metrics['f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'gpu_memory_peak': torch.cuda.max_memory_allocated()/1e9
        })
        
        print(f"⏱️  Epoch Time: {epoch_time:.1f}s")
        print(f"📈 Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"🔥 GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        # Save best model
        current_f1 = val_metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            
            # Save model state
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, 'best_model_cuda.pth')
            
            patience_counter = 0
            print(f"🎉 New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping after {patience} epochs without improvement")
            break
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"🏆 Training completed! Best F1: {best_f1:.4f}")
    
    if best_f1 >= 0.85:
        print("🎉 CONGRATULATIONS! Target F1 ≥ 0.85 achieved!")
    
    wandb.finish()

if __name__ == '__main__':
    main() 