"""
ResNet50 Classification — Student Behavior Detection.

Balanced crop dataset ustida fine-tuning.
Training tugagach YOLO formatiga o'xshash to'liq hisobot chiqaradi:
  - results.csv          (epoch bo'yicha metrikalar)
  - args.yaml            (training parametrlari)
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - training_curves.png  (loss va accuracy grafigi)
  - per_class_accuracy.png
  - val_results.json     (batafsil validation natijalari)
  - best.pt / last.pt

Ishlatish:
  python train_resnet50.py
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / "balanced_dataset"
MODELS_DIR = TRAINING_DIR / "models"
RUNS_DIR = TRAINING_DIR / "runs" / "resnet50"

NUM_CLASSES = 7
TOTAL_EPOCHS = 6
BATCH_SIZE = 64
IMAGE_SIZE = 224
LR = 0.001
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "bow-head", "focus", "hand-raising", "read",
    "standing", "turn-head", "write",
]

ATTENTIVE = {"focus", "hand-raising", "read", "write"}
DISTRACTED = {"bow-head", "standing", "turn-head"}


def log(msg: str):
    print(msg, flush=True)


def format_time(s):
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    return f"{m}m {sec}s" if m > 0 else f"{sec}s"


# ──────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.015),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def class_weights(dataset) -> torch.Tensor:
    counts = [0] * NUM_CLASSES
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    total = sum(counts)
    w = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(w, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
def create_model() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#0f0f1a",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "text.color": "#ccc",
    "grid.color": "#222",
    "font.size": 11,
})


def plot_confusion_matrix(y_true, y_pred, save_dir: Path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues" if not normalize else "YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ylabel="Haqiqiy", xlabel="Bashorat",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)

    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    name = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    fig.savefig(save_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {name}")


def plot_training_curves(history: dict, save_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", color="#6366f1", label="Train Loss", linewidth=2, markersize=5)
    ax1.plot(epochs, history["val_loss"], "o-", color="#ef4444", label="Val Loss", linewidth=2, markersize=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", color="#6366f1", label="Train Acc", linewidth=2, markersize=5)
    ax2.plot(epochs, history["val_acc"], "o-", color="#10b981", label="Val Acc", linewidth=2, markersize=5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: training_curves.png")


def plot_per_class_accuracy(class_accs: dict, save_dir: Path):
    names = list(class_accs.keys())
    accs = [class_accs[n]["accuracy"] for n in names]
    colors = ["#10b981" if n in ATTENTIVE else "#ef4444" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="none", height=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Validation Accuracy", fontweight="bold", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#10b981", label="Attentive"),
        Patch(facecolor="#ef4444", label="Distracted"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: per_class_accuracy.png")


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
def train():
    log(f"Device: {DEVICE}")
    log(f"Dataset: {DATASET_DIR}")
    log(f"Epochs: {TOTAL_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

    log(f"\nTrain: {len(train_ds):,} rasm")
    log(f"Val:   {len(val_ds):,} rasm")
    log(f"Klasslar: {train_ds.classes}")

    cw = class_weights(train_ds).to(DEVICE)
    log(f"\nClass weights:")
    for name, w in zip(train_ds.classes, cw):
        log(f"  {name:15s}: {w:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = create_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"\nModel: ResNet50 ({total_params:,} params)")

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # Output directory
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # results.csv
    results_csv = RUNS_DIR / "results.csv"
    csv_f = open(results_csv, "w")
    csv_f.write("epoch,time,train/loss,train/acc,val/loss,metrics/accuracy_top1,lr\n")

    # History for plots
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_acc = 0.0
    start_time = time.time()

    log("\n" + "=" * 70)
    log("  RESNET50 TRAINING BOSHLANDI")
    log("=" * 70)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        epoch_start_str = datetime.now().strftime("%H:%M:%S")
        log(f"\n  Epoch {epoch}/{TOTAL_EPOCHS} boshlandi: {epoch_start_str}")

        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"  Epoch {epoch}/{TOTAL_EPOCHS} [Train]",
                         bar_format="{l_bar}{bar:30}{r_bar}", leave=True, colour="blue")
        for images, labels in train_bar:
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

            train_bar.set_postfix(
                loss=f"{running_loss/(train_bar.n+1):.4f}",
                acc=f"{100.*correct/total:.1f}%",
            )

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        scheduler.step()

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        all_preds, all_labels = [], []

        val_bar = tqdm(val_loader, desc=f"  Epoch {epoch}/{TOTAL_EPOCHS} [Val]  ",
                       bar_format="{l_bar}{bar:30}{r_bar}", leave=True, colour="green")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(labels.size(0)):
                    lbl = labels[i].item()
                    class_total[lbl] += 1
                    if predicted[i] == lbl:
                        class_correct[lbl] += 1

                val_bar.set_postfix(
                    loss=f"{val_loss/(val_bar.n+1):.4f}",
                    acc=f"{100.*val_correct/val_total:.1f}%",
                )

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (TOTAL_EPOCHS - epoch)

        # Progress bar
        pct = epoch / TOTAL_EPOCHS
        filled = int(30 * pct)
        bar = "=" * filled + ">" + "." * (29 - filled)

        # Best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), RUNS_DIR / "best.pt")
            marker = " << YANGI BEST!"
        else:
            marker = ""

        torch.save(model.state_dict(), RUNS_DIR / "last.pt")

        # Log
        log(f"\n  [{bar}] {epoch}/{TOTAL_EPOCHS} ({pct*100:.0f}%)")
        log(f"  Epoch: {format_time(epoch_time)} | "
            f"O'tgan: {format_time(elapsed)} | "
            f"Qolgan: ~{format_time(remaining)}")
        log(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        log(f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{marker}")
        log(f"  Best Val Acc: {best_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        log(f"  Per-class accuracy:")
        for ci, cname in enumerate(CLASS_NAMES):
            if class_total[ci] > 0:
                c_acc = 100.0 * class_correct[ci] / class_total[ci]
                log(f"    {cname:15s}: {c_acc:5.1f}%  ({class_correct[ci]}/{class_total[ci]})")
        log("-" * 70)

        # CSV
        current_lr = scheduler.get_last_lr()[0]
        csv_f.write(f"{epoch},{elapsed:.2f},{train_loss:.5f},{train_acc/100:.5f},"
                    f"{val_loss:.5f},{val_acc/100:.5f},{current_lr:.7f}\n")
        csv_f.flush()

    csv_f.close()

    # ──────────────────────────────────────────────────────────
    # POST-TRAINING: Grafiklar va hisobotlar
    # ──────────────────────────────────────────────────────────
    total_time = format_time(time.time() - start_time)
    log("\n" + "=" * 70)
    log(f"  TRAINING TUGADI!")
    log(f"  Jami vaqt: {total_time}")
    log(f"  Best Val Accuracy: {best_acc:.2f}%")
    log("=" * 70)

    # Best modelni yuklash va yakuniy validation
    log("\n  Yakuniy validation (best.pt) ...")
    best_state = torch.load(RUNS_DIR / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_total[lbl] += 1
                if predicted[i] == lbl:
                    class_correct[lbl] += 1

    final_acc = 100.0 * sum(class_correct) / sum(class_total)

    # ── Confusion Matrix ──
    log("\n  Grafiklar chizilmoqda...")
    plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=False)
    plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=True)

    # ── Training Curves ──
    plot_training_curves(history, RUNS_DIR)

    # ── Per-Class Accuracy ──
    class_accs = {}
    for ci, cname in enumerate(CLASS_NAMES):
        if class_total[ci] > 0:
            class_accs[cname] = {
                "accuracy": 100.0 * class_correct[ci] / class_total[ci],
                "correct": class_correct[ci],
                "total": class_total[ci],
            }
    plot_per_class_accuracy(class_accs, RUNS_DIR)

    # ── val_results.json ──
    val_results = {
        "model": "ResNet50",
        "epochs_trained": TOTAL_EPOCHS,
        "total_params": total_params,
        "best_val_accuracy": round(final_acc, 2),
        "training_time": total_time,
        "per_class": class_accs,
    }
    (RUNS_DIR / "val_results.json").write_text(json.dumps(val_results, indent=2))
    log(f"  Saved: val_results.json")

    # ── args.yaml ──
    args_yaml = f"""model: ResNet50 (ImageNet V2 pretrained)
task: classify
data: balanced_dataset
epochs: {TOTAL_EPOCHS}
batch: {BATCH_SIZE}
imgsz: {IMAGE_SIZE}
optimizer: AdamW
lr0: {LR}
weight_decay: {WEIGHT_DECAY}
scheduler: CosineAnnealingLR
loss: CrossEntropyLoss (weighted)
augmentation:
  horizontal_flip: 0.5
  color_jitter: {{brightness: 0.3, contrast: 0.3, saturation: 0.3, hue: 0.015}}
  rotation: 10
classes: {NUM_CLASSES}
names:
{chr(10).join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))}
total_params: {total_params}
best_val_accuracy: {final_acc:.2f}
training_time: {total_time}
"""
    (RUNS_DIR / "args.yaml").write_text(args_yaml)
    log(f"  Saved: args.yaml")

    # ── Best modelni models/ ga ko'chirish ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best_resnet50.pt"
    shutil.copy2(RUNS_DIR / "best.pt", dest)
    log(f"\n  Model saved to: {dest}")

    # ── Yakuniy hisobot ──
    log("\n" + "=" * 70)
    log("  YAKUNIY HISOBOT")
    log("=" * 70)
    log(f"  Best Val Accuracy: {final_acc:.2f}%")
    log(f"  Training time:     {total_time}")
    log(f"  Saved files:")
    for f in sorted(RUNS_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            s = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            s = f"{size / 1024:.1f} KB"
        else:
            s = f"{size} B"
        log(f"    {f.name:40s} {s:>10s}")
    log("=" * 70)


if __name__ == "__main__":
    train()
