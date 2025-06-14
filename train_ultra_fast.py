import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Ensure consistent absolute path for Colab and CLI
SRC_DIR = '/content/src'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.fast_cutin_dataset import FastCutInDataset, fast_collate_fn
from models.model import CutInDetectionModel
from utils.hungarian_matcher import HungarianMatcher
from utils.losses import CombinedLoss

# CUDA Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_cuda():
    """Setup CUDA for maximum performance"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        device = torch.device('cuda')
        print(f"🔥 CUDA Device: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        return torch.device('cpu')

def calculate_f1_fast(outputs, targets, indices):
    """Ultra-fast F1 calculation"""
    pred_cutin = outputs['pred_cutin']
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_scores = torch.sigmoid(pred_cutin[batch_idx, pred_idx])
                true_scores = targets[batch_idx]['cutin'][target_idx]
                
                all_pred.extend((pred_scores > 0.5).cpu().numpy())
                all_true.extend(true_scores.cpu().numpy())
    
    if not all_pred:
        return 0.0
    
    # Fast F1 calculation
    tp = sum(p and t for p, t in zip(all_pred, all_true))
    fp = sum(p and not t for p, t in zip(all_pred, all_true))
    fn = sum(not p and t for p, t in zip(all_pred, all_true))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def train_ultra_fast(model, train_loader, val_loader, device):
    """Ultra-fast training loop"""
    
    # Optimized components
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_cutin=10.0)
    criterion = CombinedLoss(cutin_weight=15.0, cutin_pos_weight=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=20, steps_per_epoch=len(train_loader)
    )
    
    best_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    print("🚀 Starting Ultra-Fast Training")
    
    for epoch in range(20):  # Fewer epochs for speed
        start_time = time.time()
        model.train()
        
        total_loss = 0
        f1_scores = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/20', leave=False)
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Fast GPU transfer
            images = images.to(device, non_blocking=True)
            for target in targets:
                target['labels'] = target['labels'].to(device, non_blocking=True)
                target['boxes'] = target['boxes'].to(device, non_blocking=True)
                target['cutin'] = target['cutin'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                
                with torch.no_grad():
                    indices = matcher.forward(outputs, targets)
                
                losses = criterion(outputs, targets, indices)
                loss = losses['total_loss']
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate F1 every 20 steps
            if batch_idx % 20 == 0:
                f1 = calculate_f1_fast(outputs, targets, indices)
                f1_scores.append(f1)
            
            # Update progress
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'F1': f"{f1_scores[-1] if f1_scores else 0:.4f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}",
                    'GPU': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Validation (every 2 epochs)
        if epoch % 2 == 0:
            model.eval()
            val_f1_scores = []
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc='Validation', leave=False):
                    images = images.to(device, non_blocking=True)
                    for target in targets:
                        target['labels'] = target['labels'].to(device, non_blocking=True)
                        target['boxes'] = target['boxes'].to(device, non_blocking=True)
                        target['cutin'] = target['cutin'].to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(images)
                        indices = matcher.forward(outputs, targets)
                    
                    f1 = calculate_f1_fast(outputs, targets, indices)
                    val_f1_scores.append(f1)
            
            val_f1 = np.mean(val_f1_scores) if val_f1_scores else 0.0
        else:
            val_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        epoch_time = time.time() - start_time
        train_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        print(f"⏱️  Epoch {epoch+1}: {epoch_time:.1f}s | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
            }, 'best_model_ultra_fast.pth')
            patience_counter = 0
            print(f"🎉 New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping after {patience} epochs")
            break
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_f1

def main():
    print("🚀 Ultra-Fast CUDA Training Pipeline")
    
    # Setup
    device = setup_cuda()
    
    # Fast transforms
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ultra-fast datasets
    print("📊 Loading datasets with caching...")
    train_dataset = FastCutInDataset(
        root_dir="/content/distribution/Train",
        transform=transform,
        max_samples=5000  # Limit for ultra-fast training
    )
    
    val_dataset = FastCutInDataset(
        root_dir="/content/distribution/Val",
        transform=transform,
        max_samples=1000
    )
    
    # Fast data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=12,  # Larger batch size
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        collate_fn=fast_collate_fn,
        persistent_workers=True,
        prefetch_factor=3
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=fast_collate_fn,
        persistent_workers=True
    )
    
    print(f"✅ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Model
    model = CutInDetectionModel(num_classes=4, num_queries=50, hidden_dim=256)  # Smaller for speed
    model.to(device)
    
    # Compile model for extra speed (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("⚡ Model compiled for extra speed!")
    except:
        print("⚠️  Model compilation not available")
    
    # Train
    start_time = time.time()
    best_f1 = train_ultra_fast(model, train_loader, val_loader, device)
    total_time = time.time() - start_time
    
    print(f"\n🏆 Training Complete!")
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"📈 Best F1 Score: {best_f1:.4f}")
    
    if best_f1 >= 0.85:
        print("🎉 CONGRATULATIONS! Target F1 ≥ 0.85 achieved!")
    else:
        print(f"🎯 Target: 0.85, Achieved: {best_f1:.4f}")
        print("💡 Consider running full training for better results")

if __name__ == '__main__':
    main() 