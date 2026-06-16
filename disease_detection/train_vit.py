"""
disease_detection/train_vit.py
================================
Vision Transformer (ViT-Small/16) for PlantVillage Disease Detection
=====================================================================
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Paths & Hyperparameters ──────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "plantvillage"
MODEL_DIR = BASE_DIR / "core" / "ml_models" / "vit_disease_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VIT_CHECKPOINT   = "WinKawaks/vit-small-patch16-224"
IMG_SIZE         = 224
BATCH_SIZE       = 16
EPOCHS_HEAD      = 5
EPOCHS_FINETUNE  = 10
LR_HEAD          = 2e-4
LR_FINETUNE      = 5e-5
WEIGHT_DECAY     = 0.01
NUM_WORKERS      = 2
SEED             = 42

# ── Helper Functions ──────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, freeze_backbone=False):
    model.train()
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = ("classifier" in name)
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 50 == 0:
            print(f"     Batch {batch_idx+1}/{len(loader)} | Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100*correct/total:.1f}%")
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    top3_correct = 0
    # Small subset for top-3 to save time
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images)
        top3 = outputs.logits.topk(3, dim=1).indices
        top3_correct += sum(labels[i].item() in top3[i].tolist() for i in range(len(labels)))
    
    return (total_loss / len(loader), correct / total, top3_correct / total, all_preds, all_labels)

class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label

# ── Windows Safe Main Wrapper ────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  AgriBot — ViT Disease Detection Training")
    print("=" * 60)

    print("\n[1/6] Initializing...")
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {DEVICE}")

    # [2/6] Pipeline
    print("\n[2/6] Building data pipeline...")
    processor = ViTImageProcessor.from_pretrained(VIT_CHECKPOINT)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    full_dataset = datasets.ImageFolder(str(DATA_DIR))
    CLASS_NAMES = full_dataset.classes
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_idx, val_idx = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

    train_loader = DataLoader(SubsetWithTransform(full_dataset, train_idx.indices, train_transform), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(SubsetWithTransform(full_dataset, val_idx.indices, val_transform), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # [3/6] Load Model
    print(f"\n[3/6] Loading ViT...")
    model = ViTForImageClassification.from_pretrained(VIT_CHECKPOINT, num_labels=len(CLASS_NAMES), ignore_mismatched_sizes=True).to(DEVICE)

    # [4/6 & 5/6] Training Phases
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0
    best_model_path = str(MODEL_DIR / "best_model")

    # Phase 1
    optimizer1 = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    for epoch in range(EPOCHS_HEAD):
        train_epoch(model, train_loader, optimizer1, criterion, DEVICE, freeze_backbone=True)
        _, val_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_path)

    # Phase 2
    for param in model.parameters(): param.requires_grad = True
    optimizer2 = AdamW(model.parameters(), lr=LR_FINETUNE)
    for epoch in range(EPOCHS_FINETUNE):
        train_epoch(model, train_loader, optimizer2, criterion, DEVICE)
        _, val_acc, _, all_preds, all_labels = evaluate(model, val_loader, criterion, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_path)

    # [6/6] Final Saving & Reporting (Your requested lines are here)
    print(f"\n[6/6] Saving final model and evaluation report...")
    config = {"model_type": "vit", "class_names": CLASS_NAMES, "best_acc": round(best_val_acc * 100, 2)}
    config_path = MODEL_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"   FINAL RESULTS")
    print(f"{'='*60}")
    print(f"   Best Val Acc   : {best_val_acc*100:.2f}%")

    print(f"\n✅  Model saved  → {best_model_path}/")
    print(f"✅  Config saved → {config_path}")
    print(f"\n🚀  Next step: update views.py to use ViT")
    print(f"    (Already done if you replaced views.py with the ViT version)")