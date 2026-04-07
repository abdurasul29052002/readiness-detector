"""
PoseCNN Fusion — CNN + Pose Estimation Fusion for Student Behavior Detection.

Yangi algoritm: ResNet50 vizual featurelari (2048-dim) + MediaPipe Pose
landmarklari (33×4=132-dim) birlashtiriladi. Bu domain-specific fusion
yondashuvi "qo'l ko'tarish", "yozish" kabi harakatlarni faqat rasmdan emas,
skeletal (poza) ma'lumotidan ham aniqlaydi.

Arxitektura:
  Input Image
    ├── ResNet50 backbone ──► 2048-dim visual features
    ├── MediaPipe Pose ──────► 132-dim pose features (33 landmarks × 4)
    └── Fusion: concat(2048 + 132) ──► FC(1024) ──► FC(7)

Training tugagach to'liq hisobot:
  - results.csv, args.yaml, confusion_matrix.png, training_curves.png
  - per_class_accuracy.png, val_results.json, best.pt / last.pt

Ishlatish:
  python train_fusion.py
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

# MediaPipe pose extraction
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / "balanced_dataset"
MODELS_DIR = TRAINING_DIR / "models"
RUNS_DIR = TRAINING_DIR / "runs" / "fusion"
POSE_CACHE_DIR = TRAINING_DIR / "pose_cache"
POSE_MODEL_PATH = TRAINING_DIR / "pose_landmarker_lite.task"

NUM_CLASSES = 7
TOTAL_EPOCHS = 6
BATCH_SIZE = 64
IMAGE_SIZE = 224
LR = 0.001
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pose features: 33 landmarks × 4 (x, y, z, visibility)
NUM_LANDMARKS = 33
POSE_FEAT_DIM = NUM_LANDMARKS * 4  # 132

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
# POSE EXTRACTION
# ──────────────────────────────────────────────────────────────
def create_pose_detector():
    """MediaPipe Pose Landmarker yaratish."""
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_features(detector, image_path: str) -> np.ndarray:
    """Rasmdan 132-dim pose feature vector olish."""
    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        # Poza topilmasa nol vektor
        return np.zeros(POSE_FEAT_DIM, dtype=np.float32)

    landmarks = result.pose_landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(features, dtype=np.float32)


def cache_pose_features(split: str):
    """Dataset uchun pose featurelarni oldindan hisoblash va cache qilish."""
    cache_dir = POSE_CACHE_DIR / split
    split_dir = DATASET_DIR / split

    # Agar cache allaqachon to'liq bo'lsa, o'tkazib yuborish
    if cache_dir.exists():
        cached_count = sum(1 for _ in cache_dir.rglob("*.npy"))
        total_images = sum(1 for _ in split_dir.rglob("*.jpg")) + \
                       sum(1 for _ in split_dir.rglob("*.png")) + \
                       sum(1 for _ in split_dir.rglob("*.jpeg"))
        if cached_count >= total_images > 0:
            log(f"  {split}: cache tayyor ({cached_count:,} fayl)")
            return

    log(f"  {split}: pose featurelar hisoblanmoqda...")
    detector = create_pose_detector()

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_cache = cache_dir / class_dir.name
        class_cache.mkdir(parents=True, exist_ok=True)

        images = sorted(class_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        bar = tqdm(images, desc=f"    {class_dir.name}", leave=True,
                   bar_format="{l_bar}{bar:25}{r_bar}")
        for img_path in bar:
            npy_path = class_cache / (img_path.stem + ".npy")
            if npy_path.exists():
                continue
            try:
                features = extract_pose_features(detector, str(img_path))
                np.save(npy_path, features)
            except Exception:
                np.save(npy_path, np.zeros(POSE_FEAT_DIM, dtype=np.float32))

    detector.close()


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """Rasm + Pose feature qaytaradigan dataset."""

    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.split = split

        # ImageFolder orqali rasm ro'yxatini olish
        split_dir = DATASET_DIR / split
        self.samples = []  # [(image_path, label), ...]
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Rasm
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Pose features (cache dan)
        img_name = Path(img_path)
        npy_path = POSE_CACHE_DIR / self.split / img_name.parent.name / (img_name.stem + ".npy")
        if npy_path.exists():
            pose = np.load(npy_path)
        else:
            pose = np.zeros(POSE_FEAT_DIM, dtype=np.float32)

        pose_tensor = torch.from_numpy(pose)

        return image, pose_tensor, label


# ──────────────────────────────────────────────────────────────
# TRANSFORMS
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
# MODEL: PoseCNN Fusion
# ──────────────────────────────────────────────────────────────
class PoseCNNFusion(nn.Module):
    """
    ResNet50 visual features (2048) + Pose features (132)
    → Fusion → Classification (7 classes)

    Ilmiy yangilik: CNN va skeletal pose ma'lumotini birlashtirib,
    o'quvchi xatti-harakatini aniqroq tasniflash.
    """

    def __init__(self, num_classes: int = 7, pose_dim: int = POSE_FEAT_DIM):
        super().__init__()

        # Visual branch: ResNet50 pretrained (FC olib tashlangan)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)
        self.visual_dim = 2048

        # Pose branch: MLP
        self.pose_branch = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pose_out_dim = 128

        # Fusion head
        fusion_dim = self.visual_dim + self.pose_out_dim  # 2048 + 128 = 2176
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, pose):
        # Visual features
        v = self.visual_backbone(image)  # (B, 2048, 1, 1)
        v = v.flatten(1)                 # (B, 2048)

        # Pose features
        p = self.pose_branch(pose)       # (B, 128)

        # Fusion
        fused = torch.cat([v, p], dim=1)  # (B, 2176)
        out = self.classifier(fused)      # (B, 7)
        return out


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

    title = "PoseCNN Fusion — Confusion Matrix"
    if normalize:
        title += " (Normalized)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    name = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    fig.savefig(save_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {name}")


def plot_training_curves(history: dict, save_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "o-", color="#6366f1", label="Train Loss", linewidth=2, markersize=3)
    ax1.plot(epochs, history["val_loss"], "o-", color="#ef4444", label="Val Loss", linewidth=2, markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("PoseCNN Fusion — Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", color="#6366f1", label="Train Acc", linewidth=2, markersize=3)
    ax2.plot(epochs, history["val_acc"], "o-", color="#10b981", label="Val Acc", linewidth=2, markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("PoseCNN Fusion — Accuracy", fontweight="bold")
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
    ax.set_title("PoseCNN Fusion — Per-Class Accuracy", fontweight="bold", fontsize=14)
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
# TRAINING
# ──────────────────────────────────────────────────────────────
def train():
    log(f"Device: {DEVICE}")
    log(f"Dataset: {DATASET_DIR}")
    log(f"Epochs: {TOTAL_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    log(f"Model: PoseCNN Fusion (ResNet50 + Pose Landmarks)")

    # ── Pose featurelarni cache qilish ──
    log("\n" + "=" * 70)
    log("  POSE FEATURE EXTRACTION")
    log("=" * 70)
    cache_pose_features("train")
    cache_pose_features("val")

    # ── Dataset ──
    train_tf, val_tf = get_transforms()
    train_ds = FusionDataset("train", transform=train_tf)
    val_ds = FusionDataset("val", transform=val_tf)

    log(f"\nTrain: {len(train_ds):,} rasm")
    log(f"Val:   {len(val_ds):,} rasm")

    cw = class_weights(train_ds).to(DEVICE)
    log(f"\nClass weights:")
    for name, w in zip(CLASS_NAMES, cw):
        log(f"  {name:15s}: {w:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # ── Model ──
    model = PoseCNNFusion(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel: PoseCNN Fusion ({total_params:,} params, {trainable_params:,} trainable)")

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

    # Output directory
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # results.csv
    results_csv = RUNS_DIR / "results.csv"
    csv_f = open(results_csv, "w")
    csv_f.write("epoch,time,train/loss,train/acc,val/loss,metrics/accuracy_top1,lr\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_acc = 0.0
    start_time = time.time()

    log("\n" + "=" * 70)
    log("  POSECNN FUSION TRAINING BOSHLANDI")
    log("=" * 70)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        log(f"\n  Epoch {epoch}/{TOTAL_EPOCHS} boshlandi: {datetime.now().strftime('%H:%M:%S')}")

        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"  Epoch {epoch}/{TOTAL_EPOCHS} [Train]",
                         bar_format="{l_bar}{bar:30}{r_bar}", leave=True, colour="blue")
        for images, poses, labels in train_bar:
            images = images.to(DEVICE)
            poses = poses.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, poses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(
                loss=f"{running_loss / (train_bar.n + 1):.4f}",
                acc=f"{100. * correct / total:.1f}%",
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
            for images, poses, labels in val_bar:
                images = images.to(DEVICE)
                poses = poses.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images, poses)
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
                    loss=f"{val_loss / (val_bar.n + 1):.4f}",
                    acc=f"{100. * val_correct / val_total:.1f}%",
                )

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

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
        log(f"\n  [{bar}] {epoch}/{TOTAL_EPOCHS} ({pct * 100:.0f}%)")
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
        csv_f.write(f"{epoch},{elapsed:.2f},{train_loss:.5f},{train_acc / 100:.5f},"
                    f"{val_loss:.5f},{val_acc / 100:.5f},{current_lr:.7f}\n")
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

    # Best modelni models/ ga ko'chirish
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best_fusion.pt"
    shutil.copy2(RUNS_DIR / "best.pt", dest)
    log(f"\n  Model saved to: {dest}")

    # Yakuniy validation (best.pt)
    log("\n  Yakuniy validation (best.pt) ...")
    best_state = torch.load(RUNS_DIR / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    val_bar = tqdm(val_loader, desc="  Final validation",
                   bar_format="{l_bar}{bar:30}{r_bar}", leave=True, colour="green")
    with torch.no_grad():
        for images, poses, labels in val_bar:
            images = images.to(DEVICE)
            poses = poses.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, poses)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_total[lbl] += 1
                if predicted[i] == lbl:
                    class_correct[lbl] += 1

    final_acc = 100.0 * sum(class_correct) / sum(class_total)

    # val_results.json
    class_accs = {}
    for ci, cname in enumerate(CLASS_NAMES):
        if class_total[ci] > 0:
            class_accs[cname] = {
                "accuracy": round(100.0 * class_correct[ci] / class_total[ci], 2),
                "correct": class_correct[ci],
                "total": class_total[ci],
            }

    val_results = {
        "model": "PoseCNN Fusion (ResNet50 + Pose Landmarks)",
        "epochs_trained": TOTAL_EPOCHS,
        "total_params": total_params,
        "best_val_accuracy": round(final_acc, 2),
        "training_time": total_time,
        "architecture": {
            "visual_branch": "ResNet50 (ImageNet V2) → 2048-dim",
            "pose_branch": f"MediaPipe Pose → {POSE_FEAT_DIM}-dim → MLP → 128-dim",
            "fusion": "Concat(2048 + 128) → FC(512) → FC(7)",
        },
        "per_class": class_accs,
    }
    (RUNS_DIR / "val_results.json").write_text(json.dumps(val_results, indent=2))
    log(f"  Saved: val_results.json")

    # args.yaml
    args_yaml = f"""model: PoseCNN Fusion (ResNet50 + Pose Landmarks)
algorithm: CNN + Pose Estimation Fusion
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
architecture:
  visual_branch: ResNet50 (ImageNet V2 pretrained) -> 2048-dim
  pose_branch: MediaPipe Pose (33 landmarks x 4) -> MLP -> 128-dim
  fusion: Concat(2048 + 128) -> FC(512) -> Dropout(0.4) -> FC(7)
total_params: {total_params}
best_val_accuracy: {final_acc:.2f}
training_time: {total_time}
"""
    (RUNS_DIR / "args.yaml").write_text(args_yaml)
    log(f"  Saved: args.yaml")

    # Grafiklar
    log("\n  Grafiklar chizilmoqda...")
    try:
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=False)
        plot_confusion_matrix(all_labels, all_preds, RUNS_DIR, normalize=True)
        plot_training_curves(history, RUNS_DIR)
        plot_per_class_accuracy(class_accs, RUNS_DIR)
    except Exception as e:
        log(f"  WARNING: Grafik chizishda xato: {e}")

    # Yakuniy hisobot
    log("\n" + "=" * 70)
    log("  YAKUNIY HISOBOT — PoseCNN Fusion")
    log("=" * 70)
    log(f"  Best Val Accuracy: {final_acc:.2f}% (Baseline ResNet50: 69.6%)")
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
