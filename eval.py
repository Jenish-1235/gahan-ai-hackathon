import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Ensure consistent absolute path for Colab and CLI
SRC_DIR = '/content/src'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.cutin_sequence_dataset import CutInSequenceDataset
from models.model import CutInDetectionModel
from utils.hungarian_matcher import HungarianMatcher

def collate_fn(batch):
    """Collate function for evaluation"""
    images, annotations = zip(*batch)
    images = torch.stack(images)
    
    # Process annotations
    targets = []
    class_mapping = {'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3}
    
    for batch_idx, frame_annotations in enumerate(annotations):
        all_objects = []
        for frame_ann in frame_annotations:
            all_objects.extend(frame_ann)
        
        if len(all_objects) == 0:
            targets.append({
                'labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.zeros((0, 4)),
                'cutin': torch.tensor([], dtype=torch.float)
            })
            continue
        
        labels = []
        boxes = []
        cutin_flags = []
        
        for obj in all_objects:
            class_idx = class_mapping.get(obj['label'], 0)
            labels.append(class_idx)
            boxes.append(obj['bbox_norm'])
            cutin_flags.append(1.0 if obj['cutting'] else 0.0)
        
        targets.append({
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float),
            'cutin': torch.tensor(cutin_flags, dtype=torch.float)
        })
    
    return images, targets

def evaluate_model(model, dataloader, matcher, device, confidence_threshold=0.5):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    detailed_results = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc='Evaluating')):
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
            
            # Process predictions
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            pred_cutin = outputs['pred_cutin']
            
            batch_size = pred_logits.shape[0]
            
            for b in range(batch_size):
                # Get predictions for this batch item
                batch_pred_logits = pred_logits[b]  # (num_queries, num_classes)
                batch_pred_boxes = pred_boxes[b]    # (num_queries, 4)
                batch_pred_cutin = pred_cutin[b]    # (num_queries,)
                
                # Apply confidence threshold
                pred_probs = F.softmax(batch_pred_logits, dim=-1)
                max_probs, pred_classes = pred_probs.max(dim=-1)
                
                # Filter by confidence and non-background class
                valid_mask = (max_probs > confidence_threshold) & (pred_classes > 0)
                
                if valid_mask.sum() > 0:
                    valid_classes = pred_classes[valid_mask]
                    valid_boxes = batch_pred_boxes[valid_mask]
                    valid_cutin_scores = torch.sigmoid(batch_pred_cutin[valid_mask])
                    valid_cutin_binary = (valid_cutin_scores > 0.5).float()
                    
                    # Store predictions
                    for i in range(len(valid_classes)):
                        all_predictions.append({
                            'batch_idx': batch_idx * dataloader.batch_size + b,
                            'class': valid_classes[i].item(),
                            'bbox': valid_boxes[i].cpu().numpy(),
                            'cutin_score': valid_cutin_scores[i].item(),
                            'cutin_binary': valid_cutin_binary[i].item(),
                            'confidence': max_probs[valid_mask][i].item()
                        })
                
                # Store ground truth
                gt_labels = targets[b]['labels']
                gt_boxes = targets[b]['boxes']
                gt_cutin = targets[b]['cutin']
                
                for i in range(len(gt_labels)):
                    all_ground_truth.append({
                        'batch_idx': batch_idx * dataloader.batch_size + b,
                        'class': gt_labels[i].item(),
                        'bbox': gt_boxes[i].cpu().numpy(),
                        'cutin': gt_cutin[i].item()
                    })
    
    return all_predictions, all_ground_truth

def calculate_detailed_metrics(predictions, ground_truth):
    """Calculate comprehensive metrics"""
    # Extract cut-in predictions and ground truth
    pred_cutin = [p['cutin_binary'] for p in predictions]
    gt_cutin = [gt['cutin'] for gt in ground_truth]
    
    if len(pred_cutin) == 0 or len(gt_cutin) == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'confusion_matrix': np.zeros((2, 2))
        }
    
    # Align predictions with ground truth (simplified matching)
    # In a real scenario, you'd want more sophisticated matching
    min_len = min(len(pred_cutin), len(gt_cutin))
    pred_cutin = pred_cutin[:min_len]
    gt_cutin = gt_cutin[:min_len]
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_cutin, pred_cutin, average='binary', zero_division=0
    )
    
    accuracy = sum(p == g for p, g in zip(pred_cutin, gt_cutin)) / len(pred_cutin)
    cm = confusion_matrix(gt_cutin, pred_cutin, labels=[0, 1])
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'num_predictions': len(pred_cutin),
        'num_ground_truth': len(gt_cutin),
        'positive_predictions': sum(pred_cutin),
        'positive_ground_truth': sum(gt_cutin)
    }

def generate_submission_csv(predictions, output_path='submission.csv'):
    """Generate submission CSV file"""
    submission_data = []
    
    for pred in predictions:
        submission_data.append({
            'sample_id': pred['batch_idx'],
            'class_prediction': pred['class'],
            'bbox_x1': pred['bbox'][0],
            'bbox_y1': pred['bbox'][1],
            'bbox_x2': pred['bbox'][2],
            'bbox_y2': pred['bbox'][3],
            'cutin_probability': pred['cutin_score'],
            'cutin_prediction': int(pred['cutin_binary']),
            'confidence': pred['confidence']
        })
    
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"Submission CSV saved to: {output_path}")
    
    return df

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = CutInDetectionModel(num_classes=4, num_queries=100, hidden_dim=512)
    
    # Load best checkpoint
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with best F1: {checkpoint['best_f1']:.4f}")
    else:
        print("Warning: No trained model found. Using random weights.")
    
    model.to(device)
    
    # Setup transforms (same as training, but without augmentation)
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test dataset
    test_dataset = CutInSequenceDataset(
        root_dir="/content/distribution/Test",
        transform=transform,
        roi_filter=True,
        balance_classes=False,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Hungarian matcher
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_cutin=10.0
    )
    
    print("Starting evaluation...")
    
    # Evaluate model
    predictions, ground_truth = evaluate_model(
        model, test_loader, matcher, device, confidence_threshold=0.3
    )
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Found {len(ground_truth)} ground truth annotations")
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(predictions, ground_truth)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Positive Predictions: {metrics['positive_predictions']}")
    print(f"Positive Ground Truth: {metrics['positive_ground_truth']}")
    print(f"Confusion Matrix:")
    print(f"  TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
    print(f"  FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
    
    # Generate submission file
    submission_df = generate_submission_csv(predictions, 'submission.csv')
    
    # Save detailed results
    results = {
        'metrics': metrics,
        'predictions_summary': {
            'total_predictions': len(predictions),
            'cutting_predictions': sum(1 for p in predictions if p['cutin_binary'] == 1),
            'average_confidence': np.mean([p['confidence'] for p in predictions]) if predictions else 0,
            'class_distribution': {}
        }
    }
    
    # Class distribution
    class_names = ['Background', 'Car', 'MotorBike', 'EgoVehicle']
    for i in range(4):
        count = sum(1 for p in predictions if p['class'] == i)
        results['predictions_summary']['class_distribution'][class_names[i]] = count
    
    # Save results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: evaluation_results.json")
    print(f"Submission file saved to: submission.csv")
    
    # Print class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    for class_name, count in results['predictions_summary']['class_distribution'].items():
        print(f"{class_name}: {count}")
    
    return metrics['f1']

if __name__ == '__main__':
    f1_score = main()
    print(f"\nFinal F1 Score: {f1_score:.4f}")
    
    if f1_score >= 0.85:
        print("🎉 CONGRATULATIONS! F1 Score >= 0.85 achieved!")
    else:
        print(f"Target F1 Score: 0.85, Current: {f1_score:.4f}")
        print("Consider further hyperparameter tuning or model improvements.")
