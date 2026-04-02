"""
YOLOv8-cls modelni balanced crop dataset ustida o'qitish.
Class weights bilan — kam klasslar uchun ko'proq og'irlik.

Ishlatish:
  1. Avval: python balance_dataset.py
  2. Keyin: python train_classifier.py
"""

import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAINING_DIR / "balanced_dataset"
MODELS_DIR = TRAINING_DIR / "models"
TOTAL_EPOCHS = 50


def count_class_distribution() -> dict[str, int]:
    """Har bir klassdagi rasm sonini hisoblaydi."""
    train_dir = DATASET_DIR / "train"
    counts = {}
    for cls_dir in sorted(train_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            counts[cls_dir.name] = count
    return counts


def calculate_class_weights(counts: dict[str, int]) -> list[float]:
    """
    Inverse frequency class weights hisoblash.
    Kam klassga ko'proq og'irlik beradi.
    """
    total = sum(counts.values())
    n_classes = len(counts)
    weights = []
    for cls_name in sorted(counts.keys()):
        # weight = total / (n_classes * count)
        w = total / (n_classes * counts[cls_name])
        weights.append(round(w, 4))
    return weights


class ClassifierProgress:
    """Classification training progress ni CLI da ko'rsatish."""

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = None
        self.best_acc = 0

    def on_train_start(self, trainer):
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print(f"  CLASSIFICATION TRAINING BOSHLANDI | {self.total_epochs} epoch | GPU: {trainer.device}")
        print("=" * 70)

    def on_train_epoch_start(self, trainer):
        self.epoch_start = time.time()
        self.current_epoch = trainer.epoch + 1

    def on_train_epoch_end(self, trainer):
        epoch_time = time.time() - self.epoch_start
        elapsed = time.time() - self.start_time
        remaining = (elapsed / self.current_epoch) * (self.total_epochs - self.current_epoch)

        # Progress bar
        pct = self.current_epoch / self.total_epochs
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Loss
        loss = trainer.loss.item() if trainer.loss is not None else 0

        print(f"\n  [{bar}] {self.current_epoch}/{self.total_epochs} ({pct*100:.0f}%)")
        print(f"  Epoch: {self._fmt(epoch_time)} | O'tgan: {self._fmt(elapsed)} | Qolgan: ~{self._fmt(remaining)}")
        print(f"  Loss: {loss:.4f}")

    def on_fit_epoch_end(self, trainer):
        metrics = trainer.metrics
        if metrics:
            top1 = metrics.get("metrics/accuracy_top1", 0)
            top5 = metrics.get("metrics/accuracy_top5", 0)

            if top1 > self.best_acc:
                self.best_acc = top1
                marker = " ★ YANGI BEST!"
            else:
                marker = ""

            print(f"  Val   -> Top-1: {top1:.4f} | Top-5: {top5:.4f}{marker}")
            print(f"  Best Top-1: {self.best_acc:.4f}")
        print("-" * 70)

    def on_train_end(self, trainer):
        total_time = self._fmt(time.time() - self.start_time)
        print("\n" + "=" * 70)
        print(f"  TRAINING TUGADI!")
        print(f"  Jami vaqt: {total_time}")
        print(f"  Best Top-1 Accuracy: {self.best_acc:.4f}")
        print("=" * 70 + "\n")

    @staticmethod
    def _fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"


def train():
    # Dataset statistikasi
    print("Dataset taqsimoti:")
    counts = count_class_distribution()
    for cls_name, count in counts.items():
        print(f"  {cls_name:15s}: {count:>7,}")

    # Class weights
    weights = calculate_class_weights(counts)
    class_names = sorted(counts.keys())
    print(f"\nClass weights:")
    for name, w in zip(class_names, weights):
        print(f"  {name:15s}: {w:.4f}")

    # Eski natijalarni tozalash
    runs_dir = TRAINING_DIR / "runs" / "yolov8-cls"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    # YOLOv8s-cls pretrained modelni yuklash
    model = YOLO("yolov8s-cls.pt")

    # Class weights ni tensor qilib tayyorlash
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    def apply_class_weights(trainer):
        """Training boshlanganida loss function ga class weights qo'yish."""
        device = trainer.device
        trainer.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(device))
        print(f"\n  Class weights qo'llanildi (device: {device})")

    # Progress callback
    progress = ClassifierProgress(TOTAL_EPOCHS)
    model.add_callback("on_train_start", progress.on_train_start)
    model.add_callback("on_train_start", apply_class_weights)
    model.add_callback("on_train_epoch_start", progress.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", progress.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", progress.on_fit_epoch_end)
    model.add_callback("on_train_end", progress.on_train_end)

    # O'qitish
    results = model.train(
        data=str(DATASET_DIR),
        epochs=TOTAL_EPOCHS,
        imgsz=224,
        batch=64,
        device=0,
        workers=4,
        project=str(TRAINING_DIR / "runs"),
        name="yolov8-cls",
        exist_ok=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.0,
        fliplr=0.5,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.01,
    )

    # best.pt ni models/ ga ko'chirish
    best_pt = TRAINING_DIR / "runs" / "yolov8-cls" / "weights" / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / "best_cls.pt"
        shutil.copy2(best_pt, dest)
        print(f"\nClassification model saved to: {dest}")
    else:
        print("\nWARNING: best.pt not found!")

    return results


if __name__ == "__main__":
    train()
