import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cut_in_dataset import CutInDataset
from models.cutting_detector import CuttingDetector
from losses.criterion import DetectionCriterion
from utils.hungarian_matcher import HungarianMatcher
from utils.collate_fn import custom_collate_fn
from utils.metrics_utils import compute_cutting_metrics

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Convert dict to object for dot-access
    class Config: pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    return config

def main():
    config = load_config("/content/jenish/gahan-ai-hackathon/configs/experiment_config.yaml")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    train_dataset = CutInDataset(config.data_root, config, mode='Train', transform=transform)
    val_dataset = CutInDataset(config.data_root, config, mode='Val', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: custom_collate_fn(batch, config)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: custom_collate_fn(batch, config)
    )

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CuttingDetector(
        vit_model_name=config.vit_model_name,
        vit_output_dim=768,
        temporal_hidden_dim=config.gru_hidden_dim,
        temporal_layers=config.gru_num_layers,
        temporal_dropout=config.gru_dropout,
        temporal_attention_heads=config.temporal_attention_heads,
        use_temporal_attention=True,
        num_queries=config.num_queries,
        num_classes=config.num_classes,
        detr_hidden_dim=config.hidden_dim,
        detr_heads=config.nheads,
        detr_decoder_layers=config.num_decoder_layers
    ).to(device)

    matcher = HungarianMatcher()
    criterion = DetectionCriterion(
        num_classes=config.num_classes,
        matcher=matcher,
        ce_loss_coef=config.ce_loss_coef,
        bbox_loss_coef=config.bbox_loss_coef,
        cutting_loss_coef=config.cutting_loss_coef
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_f1 = 0.0
    for epoch in range(config.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            outputs = model(images)
            # Move targets to device if needed
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in seq[-1].items()} for seq in targets]
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
            optimizer.step()

        # Validation
        model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                all_outputs.append(outputs)
                all_targets.extend(targets)
        # Aggregate outputs for metrics
        val_outputs = {
            'pred_cutting': torch.cat([o['pred_cutting'] for o in all_outputs], dim=0)
        }
        precision, recall, f1 = compute_cutting_metrics(val_outputs, all_targets)
        print(f"Epoch {epoch}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_cutting_detector.pth")
            print("Saved new best model.")

if __name__ == "__main__":
    main()