"""
ResNet50 Classification — Student Behavior Detection (training-v2).

Balanced crop dataset (7 klass) ustida fine-tuning.
WeightedRandomSampler + weighted CrossEntropyLoss bilan qolgan
class imbalanceni qoplaydi.

Output (xuddi YOLO kabi batafsil):
  runs/resnet50/
    environment.json          — GPU/CUDA/torch info
    args.yaml                 — training parametrlari
    results.csv               — epoch bo'yicha metrikalar
    train_batch0.jpg ...      — training augmentation namunalari
    val_batch0_labels.jpg     — haqiqiy labellar
    val_batch0_pred.jpg       — model bashoratlari
    confusion_matrix.png
    confusion_matrix_normalized.png
    training_curves.png
    per_class_accuracy.png
    PR_curve.png              — per-class Precision-Recall
    F1_curve.png              — per-class F1 vs threshold
    P_curve.png               — per-class Precision vs threshold
    R_curve.png               — per-class Recall vs threshold
    summary.json              — to'liq per-class P/R/F1 (YOLO formatida)
    val_results.json
    weights/
      best.pt / last.pt
      epoch5.pt, epoch10.pt   — checkpoint'lar

Ishlatish:
  python train_resnet50.py --activation relu      # baseline
  python train_resnet50.py --activation lswish    # ilmiy yangilik
  python train_resnet50.py --activation mish      # taqqoslash
  python train_resnet50.py --activation silu      # taqqoslash
  python train_resnet50.py --activation gelu      # taqqoslash

Aktivatsiyaga qarab natijalar `runs/resnet50_<activation>/` ga yoziladi.
"""

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from activations import (
    LearnableSwish,
    LearnableSwishChannel,
    build_activation,
    count_activation_params,
    list_beta_values,
    replace_activations,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / "balanced_dataset"
MODELS_DIR = TRAINING_DIR / "models"

NUM_CLASSES = 7
TOTAL_EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = 224
LR = 0.001
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
SAVE_PERIOD = 5  # har 5 epochda checkpoint saqlash
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVATION_CHOICES = ["relu", "mish", "silu", "gelu", "lswish"]

# ImageFolder alifbo tartibida ochadi — shu tartibda yozamiz
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


def log_environment(out_dir: Path) -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(info, indent=2))
    log("=" * 60)
    log("ENVIRONMENT")
    for k, v in info.items():
        log(f"  {k}: {v}")
    log("=" * 60)
    return info


# ──────────────────────────────────────────────────────────────
# BATCH VISUALIZATION (YOLO train_batch*.jpg / val_batch*_pred.jpg kabi)
# ──────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Normalized tensor [C,H,W] -> HWC numpy [0,1]."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def save_batch_grid(images: torch.Tensor, labels: list, preds: list | None,
                    class_names: list, save_path: Path, title: str,
                    n: int = 16):
    """Batchdagi dastlabki n ta rasmni grid qilib saqlash.
       preds=None bo'lsa faqat true labellar, aks holda true+pred ko'rsatiladi."""
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = denormalize(images[i])
        ax.imshow(img)
        true_name = class_names[labels[i]]
        if preds is None:
            ax.set_title(true_name, fontsize=9, color="#ccc")
        else:
            pred_name = class_names[preds[i]]
            ok = labels[i] == preds[i]
            color = "#10b981" if ok else "#ef4444"
            ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=8, color=color)

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#ccc")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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


def class_counts(dataset) -> list[int]:
    counts = [0] * NUM_CLASSES
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    return counts


def class_weights(counts: list[int]) -> torch.Tensor:
    total = sum(counts)
    w = [total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(w, dtype=torch.float32)


def sample_weights(dataset, counts: list[int]) -> torch.Tensor:
    # Har bir sample uchun 1/count_of_its_class — kam klasslar ko'p ko'riladi
    per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    weights = [per_class[lbl] for _, lbl in dataset.samples]
    return torch.tensor(weights, dtype=torch.double)


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
def create_model(activation: str = "relu") -> tuple[nn.Module, dict]:
    """ResNet50 + tanlangan aktivatsiya. Qaytaradi: (model, activation_info)."""
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)

    info = {"name": activation, "replaced": 0, "learnable_params": 0}
    if activation != "relu":
        factory = build_activation(activation)
        info["replaced"] = replace_activations(m, factory)
        info["learnable_params"] = count_activation_params(m)
    return m, info


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

    ax1.plot(epochs, history["train_loss"], "o-", color="#6366f1", label="Train Loss", linewidth=2, markersize=5)
    ax1.plot(epochs, history["val_loss"], "o-", color="#ef4444", label="Val Loss", linewidth=2, markersize=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
# PR / F1 / P / R CURVES (YOLO-style, per-class, one-vs-rest)
# ──────────────────────────────────────────────────────────────
_CURVE_COLORS = ["#6366f1", "#10b981", "#ef4444", "#f59e0b",
                 "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"]


def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, save_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for ci, cname in enumerate(CLASS_NAMES):
        y_bin = (y_true == ci).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, ci])
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_bin, y_probs[:, ci])
        ax.plot(recall, precision, label=f"{cname} (AP={ap:.3f})",
                color=_CURVE_COLORS[ci % len(_CURVE_COLORS)], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall Curve (per class)", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "PR_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: PR_curve.png")


def plot_f1_p_r_vs_conf(y_true: np.ndarray, y_probs: np.ndarray, save_dir: Path):
    thresholds = np.linspace(0.0, 1.0, 101)
    f1_curves = np.zeros((NUM_CLASSES, len(thresholds)))
    p_curves = np.zeros((NUM_CLASSES, len(thresholds)))
    r_curves = np.zeros((NUM_CLASSES, len(thresholds)))

    for ci in range(NUM_CLASSES):
        y_bin = (y_true == ci).astype(int)
        for ti, t in enumerate(thresholds):
            y_pred = (y_probs[:, ci] >= t).astype(int)
            tp = int(((y_pred == 1) & (y_bin == 1)).sum())
            fp = int(((y_pred == 1) & (y_bin == 0)).sum())
            fn = int(((y_pred == 0) & (y_bin == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            p_curves[ci, ti] = p
            r_curves[ci, ti] = r
            f1_curves[ci, ti] = f1

    def _plot(curves: np.ndarray, ylabel: str, title: str, fname: str,
              show_best: bool = False):
        fig, ax = plt.subplots(figsize=(9, 7))
        for ci, cname in enumerate(CLASS_NAMES):
            ax.plot(thresholds, curves[ci], label=cname,
                    color=_CURVE_COLORS[ci % len(_CURVE_COLORS)], linewidth=2)
        mean = curves.mean(axis=0)
        ax.plot(thresholds, mean, "k--", linewidth=2.5, label=f"all classes (mean)")
        if show_best:
            best_t = thresholds[mean.argmax()]
            ax.axvline(best_t, color="white", alpha=0.4, linestyle=":",
                       label=f"best conf={best_t:.2f} (F1={mean.max():.3f})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {fname}")

    _plot(f1_curves, "F1", "F1-Confidence Curve", "F1_curve.png", show_best=True)
    _plot(p_curves, "Precision", "Precision-Confidence Curve", "P_curve.png")
    _plot(r_curves, "Recall", "Recall-Confidence Curve", "R_curve.png")


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
def train(activation: str = "relu", run_name: str | None = None,
          epochs: int = TOTAL_EPOCHS):
    if run_name is None:
        # ReLU baseline `runs/resnet50/` ga (eski natijalarni saqlash uchun),
        # boshqalar `runs/resnet50_<activation>/` ga
        run_name = "resnet50" if activation == "relu" else f"resnet50_{activation}"
    RUNS_DIR = TRAINING_DIR / "runs" / run_name
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR = RUNS_DIR / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    log_environment(RUNS_DIR)
    log(f"Device: {DEVICE}")
    log(f"Dataset: {DATASET_DIR}")
    log(f"Activation: {activation}")
    log(f"Run dir:   {RUNS_DIR}")
    log(f"Epochs: {epochs}, Batch: {BATCH_SIZE}, LR: {LR}")

    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

    # Klasslar tartibi CLASS_NAMES bilan bir xilligini tekshirish
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
    cw = class_weights(counts).to(DEVICE)
    log(f"\nKlass soni va weight:")
    for name, c, w in zip(train_ds.classes, counts, cw):
        log(f"  {name:15s}: {c:>6d} rasm | weight={w:.4f}")

    # WeightedRandomSampler — minority klasslardan ko'p tanlash
    sw = sample_weights(train_ds, counts)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # ── Sample training batchlarini saqlash (augmentation namunalari) ──
    log("\n  Sample training batchlarini saqlash...")
    train_sample_iter = iter(DataLoader(train_ds, batch_size=16, sampler=sampler,
                                        num_workers=0))
    for bi in range(3):
        imgs, lbls = next(train_sample_iter)
        save_batch_grid(imgs, lbls.tolist(), None, CLASS_NAMES,
                        RUNS_DIR / f"train_batch{bi}.jpg",
                        title=f"Training batch {bi}", n=16)
    log(f"  Saved: train_batch0.jpg, train_batch1.jpg, train_batch2.jpg")

    model, act_info = create_model(activation=activation)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"\nModel: ResNet50 ({total_params:,} params)")
    log(f"  Activation: {act_info['name']} | "
        f"replaced={act_info['replaced']} | "
        f"learnable_act_params={act_info['learnable_params']}")

    criterion = nn.CrossEntropyLoss(weight=cw)

    # Param groups: LSwish β parametrlarini weight_decay dan chiqarish
    # va ularga yuqori LR berish (kichik skalyarlar tezroq o'rganishi uchun)
    beta_params, other_params = [], []
    for m in model.modules():
        if isinstance(m, (LearnableSwish, LearnableSwishChannel)):
            beta_params += list(m.parameters())
    beta_ids = {id(p) for p in beta_params}
    other_params = [p for p in model.parameters() if id(p) not in beta_ids]

    if beta_params:
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": beta_params, "lr": LR * 5, "weight_decay": 0.0, "name": "beta"},
        ])
        log(f"  Optimizer groups: main={len(other_params)} params (lr={LR}, wd={WEIGHT_DECAY}), "
            f"beta={len(beta_params)} params (lr={LR*5}, wd=0)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    results_csv = RUNS_DIR / "results.csv"
    csv_f = open(results_csv, "w")
    csv_f.write("epoch,time,train/loss,train/acc,val/loss,metrics/accuracy_top1,lr\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_acc = 0.0
    start_time = time.time()

    log("\n" + "=" * 70)
    log("  RESNET50 TRAINING BOSHLANDI")
    log("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_start_str = datetime.now().strftime("%H:%M:%S")
        log(f"\n  Epoch {epoch}/{epochs} boshlandi: {epoch_start_str}")

        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs} [Train]",
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

        val_bar = tqdm(val_loader, desc=f"  Epoch {epoch}/{epochs} [Val]  ",
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (epochs - epoch)

        pct = epoch / epochs
        filled = int(30 * pct)
        bar = "=" * filled + ">" + "." * (29 - filled)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "best.pt")
            marker = " << YANGI BEST!"
        else:
            marker = ""

        torch.save(model.state_dict(), WEIGHTS_DIR / "last.pt")

        if epoch % SAVE_PERIOD == 0:
            torch.save(model.state_dict(), WEIGHTS_DIR / f"epoch{epoch}.pt")

        log(f"\n  [{bar}] {epoch}/{epochs} ({pct*100:.0f}%)")
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

        current_lr = scheduler.get_last_lr()[0]
        csv_f.write(f"{epoch},{elapsed:.2f},{train_loss:.5f},{train_acc/100:.5f},"
                    f"{val_loss:.5f},{val_acc/100:.5f},{current_lr:.7f}\n")
        csv_f.flush()

    csv_f.close()

    # ──────────────────────────────────────────────────────────
    # POST-TRAINING
    # ──────────────────────────────────────────────────────────
    total_time = format_time(time.time() - start_time)
    log("\n" + "=" * 70)
    log(f"  TRAINING TUGADI!")
    log(f"  Jami vaqt: {total_time}")
    log(f"  Best Val Accuracy: {best_acc:.2f}%")
    log("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"best_{run_name}.pt"
    shutil.copy2(WEIGHTS_DIR / "best.pt", dest)
    log(f"\n  Model saved to: {dest}")

    log("\n  Yakuniy validation (best.pt) ...")
    best_state = torch.load(WEIGHTS_DIR / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    all_probs = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    saved_val_batches = 0

    val_bar = tqdm(val_loader, desc="  Final validation",
                   bar_format="{l_bar}{bar:30}{r_bar}", leave=True, colour="green")
    with torch.no_grad():
        for images, labels in val_bar:
            images_dev, labels_dev = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_dev.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            for i in range(labels_dev.size(0)):
                lbl = labels_dev[i].item()
                class_total[lbl] += 1
                if predicted[i] == lbl:
                    class_correct[lbl] += 1

            # Dastlabki 3 ta batch uchun labels/pred visualization
            if saved_val_batches < 3:
                bi = saved_val_batches
                save_batch_grid(images, labels.tolist(), None, CLASS_NAMES,
                                RUNS_DIR / f"val_batch{bi}_labels.jpg",
                                title=f"Val batch {bi} — haqiqiy labellar", n=16)
                save_batch_grid(images, labels.tolist(), predicted.cpu().tolist(),
                                CLASS_NAMES,
                                RUNS_DIR / f"val_batch{bi}_pred.jpg",
                                title=f"Val batch {bi} — bashorat (T=true, P=pred)", n=16)
                saved_val_batches += 1

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    log(f"  Saved: val_batch0..2_labels.jpg / val_batch0..2_pred.jpg")

    final_acc = 100.0 * sum(class_correct) / sum(class_total)

    # ── Per-class P/R/F1 (YOLO summary.json formatida) ──
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels_np, all_preds_np, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    class_accs = {}
    per_class_metrics = {}
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

    # YOLO-uslubli summary.json
    mean_p = float(precision.mean())
    mean_r = float(recall.mean())
    mean_f1 = float(f1.mean())
    # Beta qiymatlarini saqlash (faqat lswish uchun)
    beta_values = list_beta_values(model)
    if beta_values:
        (RUNS_DIR / "lswish_betas.json").write_text(
            json.dumps([{"module": n, "beta": round(b, 6)} for n, b in beta_values],
                       indent=2)
        )
        log(f"  Saved: lswish_betas.json ({len(beta_values)} modules)")

    summary = {
        "run_name": run_name,
        "activation": act_info,
        "weights_best": str(WEIGHTS_DIR / "best.pt"),
        "weights_last": str(WEIGHTS_DIR / "last.pt"),
        "metrics_overall": {
            "accuracy_top1": round(final_acc / 100, 4),
            "precision_macro": round(mean_p, 4),
            "recall_macro": round(mean_r, 4),
            "f1_macro": round(mean_f1, 4),
        },
        "metrics_per_class": per_class_metrics,
        "training_time": total_time,
        "epochs_trained": epochs,
        "total_params": total_params,
    }
    (RUNS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"  Saved: summary.json")

    val_results = {
        "model": "ResNet50",
        "activation": act_info,
        "epochs_trained": epochs,
        "total_params": total_params,
        "best_val_accuracy": round(final_acc, 2),
        "training_time": total_time,
        "per_class": class_accs,
    }
    (RUNS_DIR / "val_results.json").write_text(json.dumps(val_results, indent=2))
    log(f"  Saved: val_results.json")

    args_yaml = f"""model: ResNet50 (ImageNet V2 pretrained)
task: classify
data: balanced_dataset
activation: {act_info['name']}
activation_replaced: {act_info['replaced']}
activation_learnable_params: {act_info['learnable_params']}
epochs: {epochs}
batch: {BATCH_SIZE}
imgsz: {IMAGE_SIZE}
optimizer: AdamW
lr0: {LR}
weight_decay: {WEIGHT_DECAY}
scheduler: CosineAnnealingLR
loss: CrossEntropyLoss (weighted)
sampler: WeightedRandomSampler
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

    log("\n  Grafiklar chizilmoqda...")
    try:
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=False)
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=True)
        plot_training_curves(history, RUNS_DIR)
        plot_per_class_accuracy(class_accs, RUNS_DIR)
        plot_pr_curve(all_labels_np, all_probs, RUNS_DIR)
        plot_f1_p_r_vs_conf(all_labels_np, all_probs, RUNS_DIR)
    except Exception as e:
        log(f"  WARNING: Grafik chizishda xato: {e}")
        log(f"  Model va natijalar saqlanib qoldi!")

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


def parse_args():
    p = argparse.ArgumentParser(description="ResNet50 trainer with configurable activation")
    p.add_argument("--activation", "-a", choices=ACTIVATION_CHOICES, default="relu",
                   help="Aktivatsiya funksiyasi (default: relu)")
    p.add_argument("--epochs", "-e", type=int, default=TOTAL_EPOCHS,
                   help=f"Epoch soni (default: {TOTAL_EPOCHS})")
    p.add_argument("--run-name", default=None,
                   help="Run nomi (default: resnet50_<activation>)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(activation=args.activation, run_name=args.run_name, epochs=args.epochs)
