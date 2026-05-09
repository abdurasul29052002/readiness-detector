"""
Ablation Study — ResNet50 klassifikator uchun gibrid balanslash
strategiyasining har bir komponentini alohida baholash.

Variantlar:
  A — Xom dataset (crop_dataset, teacher drop), no balansing/sampler/weighted
  B — Balanced dataset (undersampling), no sampler/weighted
  C — Balanced dataset + WeightedRandomSampler, no weighted loss
  D — Full hybrid (already done in train_resnet50.py, Val Acc = 91.69%)

Ishlatish:
  python train_ablation.py --mode A
  python train_ablation.py --mode B
  python train_ablation.py --mode C

Har bir variant natijalari alohida papkaga saqlanadi:
  runs/ablation_A/ ..._B/ ..._C/
    ├─ environment.json
    ├─ args.yaml
    ├─ results.csv
    ├─ summary.json            <— asosiy metrika va per-class F1
    ├─ confusion_matrix.png
    ├─ training_curves.png
    ├─ per_class_accuracy.png
    └─ weights/
        ├─ best.pt
        └─ last.pt
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CLI (faqat script sifatida ishga tushirilganda parse qilinadi)
# ──────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["A", "B", "C"],
                        help="A=xom, B=undersample, C=undersample+sampler")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


MODE = "A"          # default — test uchun
TOTAL_EPOCHS = 10

# ──────────────────────────────────────────────────────────────
# MODE CONFIG
# ──────────────────────────────────────────────────────────────
MODE_CONFIG = {
    "A": {
        "name": "Xom dataset (balanssiz, sampler'siz, vaznlarsiz)",
        "dataset_subdir": "crop_dataset",
        "use_sampler": False,
        "use_weighted_loss": False,
        "exclude_teacher": True,
    },
    "B": {
        "name": "Faqat undersampling (balanced, sampler'siz, vaznlarsiz)",
        "dataset_subdir": "balanced_dataset",
        "use_sampler": False,
        "use_weighted_loss": False,
        "exclude_teacher": False,
    },
    "C": {
        "name": "Undersampling + WeightedRandomSampler (vaznlarsiz)",
        "dataset_subdir": "balanced_dataset",
        "use_sampler": True,
        "use_weighted_loss": False,
        "exclude_teacher": False,
    },
}
CFG = MODE_CONFIG[MODE]

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / CFG["dataset_subdir"]
RUNS_DIR = TRAINING_DIR / "runs" / f"ablation_{MODE}"

NUM_CLASSES = 7
BATCH_SIZE = 64
IMAGE_SIZE = 224
LR = 0.001
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "discuss", "focus", "hand-raising", "read",
    "standing", "turn-head", "write",
]

ATTENTIVE = {"focus", "hand-raising", "read", "write", "discuss"}
DISTRACTED = {"standing", "turn-head"}


def log(msg: str):
    print(msg, flush=True)


def format_time(s):
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    return f"{m}m {sec}s" if m > 0 else f"{sec}s"


def log_environment(out_dir: Path):
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": MODE,
        "mode_description": CFG["name"],
        "dataset": CFG["dataset_subdir"],
        "use_sampler": CFG["use_sampler"],
        "use_weighted_loss": CFG["use_weighted_loss"],
        "exclude_teacher": CFG["exclude_teacher"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(info, indent=2))
    log("=" * 70)
    log(f"ABLATION MODE: {MODE}")
    log(f"  {CFG['name']}")
    log("=" * 70)
    for k, v in info.items():
        log(f"  {k}: {v}")
    log("=" * 70)


# ──────────────────────────────────────────────────────────────
# DATASET with teacher exclusion (Mode A)
# ──────────────────────────────────────────────────────────────
class ImageFolderNoTeacher(datasets.ImageFolder):
    """teacher sinfini o'tkazib yuboradigan ImageFolder."""

    def find_classes(self, directory):
        classes = sorted(
            d.name for d in os.scandir(directory)
            if d.is_dir() and d.name != "teacher"
        )
        if not classes:
            raise FileNotFoundError(f"Klasslar topilmadi: {directory}")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx


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


def class_counts(dataset) -> list[int]:
    counts = [0] * NUM_CLASSES
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    return counts


def sample_weights(dataset, counts: list[int]) -> torch.Tensor:
    per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    weights = [per_class[lbl] for _, lbl in dataset.samples]
    return torch.tensor(weights, dtype=torch.double)


def create_model() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


# ──────────────────────────────────────────────────────────────
# PLOT STYLE
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
    im = ax.imshow(cm, interpolation="nearest",
                   cmap="Blues" if not normalize else "YlOrRd")
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

    title = f"Confusion Matrix (Mode {MODE}{'Norm' if normalize else ''})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    name = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    fig.savefig(save_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, save_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#6366f1", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "o-", color="#ef4444", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Mode {MODE}: Loss", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", color="#6366f1", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "o-", color="#10b981", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Mode {MODE}: Accuracy", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(class_accs: dict, save_dir: Path):
    names = list(class_accs.keys())
    accs = [class_accs[n]["accuracy"] for n in names]
    colors = ["#10b981" if n in ATTENTIVE else "#ef4444" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="none", height=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105); ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Mode {MODE}: Per-Class Validation Accuracy",
                 fontweight="bold", fontsize=14)
    ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
def train():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR = RUNS_DIR / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    log_environment(RUNS_DIR)
    log(f"Device: {DEVICE}")
    log(f"Dataset: {DATASET_DIR}")
    log(f"Epochs: {TOTAL_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    train_tf, val_tf = get_transforms()

    # ── Dataset ──
    if CFG["exclude_teacher"]:
        train_ds = ImageFolderNoTeacher(DATASET_DIR / "train", transform=train_tf)
        val_ds = ImageFolderNoTeacher(DATASET_DIR / "val", transform=val_tf)
    else:
        train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

    if train_ds.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Klasslar tartibi mos kelmadi!\n"
            f"  ImageFolder: {train_ds.classes}\n"
            f"  Kutilgan:    {CLASS_NAMES}"
        )

    log(f"\nTrain: {len(train_ds):,} rasm")
    log(f"Val:   {len(val_ds):,} rasm")
    log(f"Klasslar: {train_ds.classes}")

    counts = class_counts(train_ds)
    log(f"\nTrain per-class soni:")
    for name, c in zip(train_ds.classes, counts):
        log(f"  {name:15s}: {c:>7d}")

    # ── Sampler (Mode C) yoki shuffle (A, B) ──
    if CFG["use_sampler"]:
        sw = sample_weights(train_ds, counts)
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                   num_workers=NUM_WORKERS, pin_memory=True,
                                   persistent_workers=True)
        log(f"  Sampler: WeightedRandomSampler YOQILDI")
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True,
                                   persistent_workers=True)
        log(f"  Sampler: O'chirildi (oddiy shuffle)")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    # ── Model ──
    model = create_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"\nModel: ResNet50 ({total_params:,} params)")

    # ── Loss: weighted (D) yoki plain (A, B, C) ──
    if CFG["use_weighted_loss"]:
        total = sum(counts)
        w = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
        cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw)
        log(f"  Loss: Weighted CrossEntropyLoss YOQILDI")
    else:
        criterion = nn.CrossEntropyLoss()
        log(f"  Loss: Oddiy CrossEntropyLoss (vaznlarsiz)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # ── Training loop ──
    results_csv = RUNS_DIR / "results.csv"
    csv_f = open(results_csv, "w")
    csv_f.write("epoch,time,train/loss,train/acc,val/loss,metrics/accuracy_top1,lr\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    start_time = time.time()

    log("\n" + "=" * 70)
    log(f"  ABLATION {MODE} TRAINING BOSHLANDI")
    log("=" * 70)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        log(f"\n  Epoch {epoch}/{TOTAL_EPOCHS} boshlandi: {datetime.now().strftime('%H:%M:%S')}")

        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"  [{MODE}] Epoch {epoch}/{TOTAL_EPOCHS} [Train]",
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

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        all_preds, all_labels = [], []

        val_bar = tqdm(val_loader, desc=f"  [{MODE}] Epoch {epoch}/{TOTAL_EPOCHS} [Val]  ",
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

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "best.pt")
            marker = " << YANGI BEST!"
        else:
            marker = ""
        torch.save(model.state_dict(), WEIGHTS_DIR / "last.pt")

        log(f"  Epoch: {format_time(epoch_time)} | O'tgan: {format_time(elapsed)}")
        log(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        log(f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{marker}")
        log(f"  Best Val Acc: {best_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        csv_f.write(f"{epoch},{elapsed:.2f},{train_loss:.5f},{train_acc/100:.5f},"
                    f"{val_loss:.5f},{val_acc/100:.5f},{scheduler.get_last_lr()[0]:.7f}\n")
        csv_f.flush()

    csv_f.close()

    # ── Yakuniy baholash best.pt bilan ──
    total_time = format_time(time.time() - start_time)
    log("\n" + "=" * 70)
    log(f"  ABLATION {MODE} TUGADI! Jami vaqt: {total_time}, Best Val Acc: {best_acc:.2f}%")
    log("=" * 70)

    log("\n  Yakuniy validation (best.pt) ...")
    best_state = torch.load(WEIGHTS_DIR / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Final val"):
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

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    final_acc = 100.0 * sum(class_correct) / sum(class_total)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels_np, all_preds_np, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    class_accs, per_class_metrics = {}, {}
    for ci, cname in enumerate(CLASS_NAMES):
        acc_val = 100.0 * class_correct[ci] / class_total[ci] if class_total[ci] > 0 else 0.0
        class_accs[cname] = {
            "accuracy": round(acc_val, 2),
            "correct": class_correct[ci],
            "total": class_total[ci],
        }
        per_class_metrics[str(ci)] = {
            "name": cname,
            "precision": round(float(precision[ci]), 4),
            "recall": round(float(recall[ci]), 4),
            "f1": round(float(f1[ci]), 4),
            "accuracy": round(acc_val / 100, 4),
            "support": int(support[ci]),
        }

    summary = {
        "run_name": f"ablation_{MODE}",
        "mode": MODE,
        "mode_description": CFG["name"],
        "config": CFG,
        "weights_best": str(WEIGHTS_DIR / "best.pt"),
        "metrics_overall": {
            "accuracy_top1": round(final_acc / 100, 4),
            "precision_macro": round(float(precision.mean()), 4),
            "recall_macro": round(float(recall.mean()), 4),
            "f1_macro": round(float(f1.mean()), 4),
        },
        "metrics_per_class": per_class_metrics,
        "training_time": total_time,
        "epochs_trained": TOTAL_EPOCHS,
        "total_params": total_params,
    }
    (RUNS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"  Saved: summary.json")

    # ── args.yaml ──
    args_yaml = f"""model: ResNet50 (ImageNet V2 pretrained)
task: classify
ablation_mode: {MODE}
mode_description: {CFG["name"]}
data: {CFG["dataset_subdir"]}
use_sampler: {CFG["use_sampler"]}
use_weighted_loss: {CFG["use_weighted_loss"]}
exclude_teacher: {CFG["exclude_teacher"]}
epochs: {TOTAL_EPOCHS}
batch: {BATCH_SIZE}
imgsz: {IMAGE_SIZE}
optimizer: AdamW
lr0: {LR}
weight_decay: {WEIGHT_DECAY}
scheduler: CosineAnnealingLR
classes: {NUM_CLASSES}
best_val_accuracy: {final_acc:.2f}
training_time: {total_time}
"""
    (RUNS_DIR / "args.yaml").write_text(args_yaml)

    # ── Plots ──
    try:
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=False)
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=True)
        plot_training_curves(history, RUNS_DIR)
        plot_per_class_accuracy(class_accs, RUNS_DIR)
    except Exception as e:
        log(f"  WARNING: Grafik chizishda xato: {e}")

    log("\n" + "=" * 70)
    log(f"  YAKUNIY HISOBOT — Mode {MODE}")
    log("=" * 70)
    log(f"  Val Accuracy:      {final_acc:.2f}%")
    log(f"  F1-macro:          {f1.mean()*100:.2f}%")
    log(f"  Precision-macro:   {precision.mean()*100:.2f}%")
    log(f"  Recall-macro:      {recall.mean()*100:.2f}%")
    log(f"  Training time:     {total_time}")
    log(f"  Natijalar: {RUNS_DIR}")
    log("=" * 70)


if __name__ == "__main__":
    _a = _parse_args()
    MODE = _a.mode
    TOTAL_EPOCHS = _a.epochs
    CFG = MODE_CONFIG[MODE]
    DATASET_DIR = TRAINING_DIR / CFG["dataset_subdir"]
    RUNS_DIR = TRAINING_DIR / "runs" / f"ablation_{MODE}"
    train()
