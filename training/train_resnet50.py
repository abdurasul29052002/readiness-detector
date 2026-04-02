"""
ResNet50 classification model — Baseline 2.
Balanced crop dataset ustida o'qitish.

YOLOv8-cls bilan solishtirish uchun shu dataset va shu epoch soni ishlatiladi.

Ishlatish:
  python train_resnet50.py
"""

import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def log(msg: str):
    print(msg, flush=True)

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / "balanced_dataset"
MODELS_DIR = TRAINING_DIR / "models"
RUNS_DIR = TRAINING_DIR / "runs" / "resnet50"

NUM_CLASSES = 7
TOTAL_EPOCHS = 5
BATCH_SIZE = 64
IMAGE_SIZE = 224
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "bow-head", "discuss", "hand-raising", "read",
    "standing", "turn-head", "write",
]


def get_transforms():
    """Train va val uchun augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.015),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def create_model() -> nn.Module:
    """ResNet50 pretrained model — oxirgi qatlam 7 klassga moslanadi."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def calculate_class_weights(dataset) -> torch.Tensor:
    """Inverse frequency class weights."""
    counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts)
    weights = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def train():
    log(f"Device: {DEVICE}")
    log(f"Dataset: {DATASET_DIR}")
    log(f"Epochs: {TOTAL_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets
    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=val_transform)

    log(f"\nTrain: {len(train_dataset)} rasm")
    log(f"Val: {len(val_dataset)} rasm")
    log(f"Klasslar: {train_dataset.classes}")

    # Class weights
    class_weights = calculate_class_weights(train_dataset).to(DEVICE)
    log(f"\nClass weights:")
    for name, w in zip(train_dataset.classes, class_weights):
        log(f"  {name:15s}: {w:.4f}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # Model
    model = create_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"\nModel: ResNet50 ({total_params:,} params)")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # Runs directory
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # results.csv — YOLOv8 formatiga o'xshash
    results_csv = RUNS_DIR / "results.csv"
    with open(results_csv, "w") as f:
        f.write("epoch,time,train/loss,metrics/accuracy_top1,val/loss,lr\n")

    best_acc = 0.0
    start_time = time.time()

    log("\n" + "=" * 70)
    log(f"  RESNET50 TRAINING BOSHLANDI")
    log("=" * 70)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 500 == 0:
                log(f"  Epoch {epoch}/{TOTAL_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.1f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                for i in range(labels.size(0)):
                    lbl = labels[i].item()
                    class_total[lbl] += 1
                    if predicted[i] == lbl:
                        class_correct[lbl] += 1

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)

        # Progress bar
        pct = epoch / TOTAL_EPOCHS
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "=" * filled + ">" + "." * (bar_len - filled - 1)

        # Best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), RUNS_DIR / "best.pt")
            marker = " << YANGI BEST!"
        else:
            marker = ""

        # Har doim oxirgi modelni saqlash
        torch.save(model.state_dict(), RUNS_DIR / "last.pt")

        log(f"\n  [{bar}] {epoch}/{TOTAL_EPOCHS} ({pct*100:.0f}%)")
        log(f"  Epoch: {format_time(epoch_time)} | O'tgan: {format_time(elapsed)} | Qolgan: ~{format_time(remaining)}")
        log(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        log(f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{marker}")
        log(f"  Best Val Acc: {best_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        log(f"  Per-class accuracy:")
        for ci, cname in enumerate(CLASS_NAMES):
            if class_total[ci] > 0:
                c_acc = 100. * class_correct[ci] / class_total[ci]
                log(f"    {cname:15s}: {c_acc:.1f}% ({class_correct[ci]}/{class_total[ci]})")
        log("-" * 70)

        # results.csv ga yozish
        current_lr = scheduler.get_last_lr()[0]
        with open(results_csv, "a") as f:
            f.write(f"{epoch},{elapsed:.2f},{train_loss:.5f},{val_acc/100:.5f},{val_loss:.5f},{current_lr:.7f}\n")

    # Yakuniy
    total_time = format_time(time.time() - start_time)
    log("\n" + "=" * 70)
    log(f"  TRAINING TUGADI!")
    log(f"  Jami vaqt: {total_time}")
    log(f"  Best Val Accuracy: {best_acc:.2f}%")
    log("=" * 70)

    # Best modelni models/ ga ko'chirish
    best_pt = RUNS_DIR / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / "best_resnet50.pt"
        shutil.copy2(best_pt, dest)
        log(f"\nModel saved to: {dest}")


if __name__ == "__main__":
    train()
