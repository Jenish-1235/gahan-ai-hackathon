import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Ensure consistent absolute path for Colab and CLI
SRC_DIR = '/content/src'
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets.cutin_sequence_dataset import CutInSequenceDataset
from models.model import CutInDetectionModel

def collate_fn(batch):
    images, annotations = zip(*batch)
    return torch.stack(images), annotations

def compute_loss(pred_class, pred_bbox, pred_cutin, targets, criterion_cls, criterion_bbox, criterion_cutin):
    # Placeholder: match predictions to targets by IoU (Hungarian matching if needed)
    # For now, assume targets are aligned with predictions
    # Add your matcher here if necessary

    cls_loss = criterion_cls(pred_class.view(-1, pred_class.size(-1)), targets['class'].view(-1))
    bbox_loss = criterion_bbox(pred_bbox.view(-1, 4), targets['bbox'].view(-1, 4))
    cutin_loss = criterion_cutin(pred_cutin.view(-1), targets['cutin'].float().view(-1))

    total = cls_loss + bbox_loss + cutin_loss
    return total, cls_loss, bbox_loss, cutin_loss

def train(model, dataloader, optimizer, device):
    model.train()
    model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.L1Loss()
    criterion_cutin = nn.BCELoss()

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)  # (B, T, C, H, W)
        # Placeholder: Convert targets to tensors
        dummy_targets = {
            'class': torch.randint(0, 4, (images.size(0), 100)).to(device),
            'bbox': torch.rand(images.size(0), 100, 4).to(device),
            'cutin': torch.randint(0, 2, (images.size(0), 100)).to(device)
        }

        optimizer.zero_grad()
        pred_class, pred_bbox, pred_cutin = model(images)
        loss, cls_loss, bbox_loss, cutin_loss = compute_loss(
            pred_class, pred_bbox, pred_cutin, dummy_targets,
            criterion_cls, criterion_bbox, criterion_cutin
        )

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, BBox: {bbox_loss.item():.4f}, CutIn: {cutin_loss.item():.4f}")

def main():
    root_dir = "/content/distribution/Train"
    transform = Compose([Resize((224,224)), ToTensor()])
    dataset = CutInSequenceDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = CutInDetectionModel()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataloader, optimizer, device)

if __name__ == '__main__':
    main()