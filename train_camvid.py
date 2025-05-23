import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from camvid_dataset import CamVidDataset
from utils import compute_iou
import numpy as np

# Load config
with open("camvid_config.yaml") as f:
    config = yaml.safe_load(f)

device = config['training']['device']
num_classes = config['model']['num_classes']

# Model
model = models.segmentation.__dict__[config['model']['name']](pretrained=config['model']['pretrained'])
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model = model.to(device)

# Dataset
train_set = CamVidDataset(
    root=config['dataset']['root'],
    image_dir=config['dataset']['image_dir'],
    mask_dir=config['dataset']['mask_dir'],
    crop_size=tuple(config['dataset']['crop_size']),
    transform=True
)
train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True,
                          num_workers=config['dataset']['num_workers'])

# Loss, Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=config['dataset']['ignore_index'])
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

# Training loop
for epoch in range(config['training']['epochs']):
    model.train()
    total_loss = 0
    iou_list = []

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # IoU calculation
        preds = torch.argmax(outputs, dim=1)
        iou = compute_iou(preds, masks, num_classes, config['dataset']['ignore_index'])
        iou_list.append(iou)

    # 统计mIoU
    iou_array = np.array(iou_list)
    mean_ious = np.nanmean(iou_array, axis=0)
    mean_miou = np.nanmean(mean_ious)

    print(f"Epoch {epoch+1}/{config['training']['epochs']}")
    print(f"Loss: {total_loss / len(train_loader):.4f}")
    print(f"Per-class IoU: {[f'{v:.3f}' if not np.isnan(v) else 'nan' for v in mean_ious]}")
    print(f"Mean IoU: {mean_miou:.4f}")

torch.save(model.state_dict(), config['training']['save_path'])
