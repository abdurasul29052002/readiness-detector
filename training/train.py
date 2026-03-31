"""
YOLOv8s modelni student behavior dataset ustida o'qitish.
O'qitilgan best.pt ni models/ papkasiga ko'chiradi.

Ishlatish:
  python train.py
"""

import shutil
import sys
import time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import callbacks

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_YAML = TRAINING_DIR / "dataset.yaml"
MODELS_DIR = TRAINING_DIR / "models"
TOTAL_EPOCHS = 50


class TrainingProgress:
    """Training progress ni CLI da chiroyli ko'rsatish."""

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.start_time = None
        self.epoch_start = None
        self.best_map50 = 0

    def on_train_start(self, trainer):
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print(f"  TRAINING BOSHLANDI | {self.total_epochs} epoch | GPU: {trainer.device}")
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

        # Loss values
        box_loss = trainer.loss_items[0].item() if trainer.loss_items is not None else 0
        cls_loss = trainer.loss_items[1].item() if trainer.loss_items is not None else 0
        dfl_loss = trainer.loss_items[2].item() if trainer.loss_items is not None else 0

        # Time formatting
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time(remaining)
        epoch_str = self._format_time(epoch_time)

        print(f"\n  [{bar}] {self.current_epoch}/{self.total_epochs} ({pct*100:.0f}%)")
        print(f"  Epoch vaqti: {epoch_str} | O'tgan: {elapsed_str} | Qolgan: ~{remaining_str}")
        print(f"  Loss  -> box: {box_loss:.4f} | cls: {cls_loss:.4f} | dfl: {dfl_loss:.4f}")

    def on_fit_epoch_end(self, trainer):
        metrics = trainer.metrics
        if metrics:
            map50 = metrics.get("metrics/mAP50(B)", 0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
            precision = metrics.get("metrics/precision(B)", 0)
            recall = metrics.get("metrics/recall(B)", 0)

            if map50 > self.best_map50:
                self.best_map50 = map50
                marker = " ★ YANGI BEST!"
            else:
                marker = ""

            print(f"  Val   -> mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | P: {precision:.4f} | R: {recall:.4f}{marker}")
            print(f"  Best mAP50: {self.best_map50:.4f}")
        print("-" * 70)

    def on_train_end(self, trainer):
        total_time = self._format_time(time.time() - self.start_time)
        print("\n" + "=" * 70)
        print(f"  TRAINING TUGADI!")
        print(f"  Jami vaqt: {total_time}")
        print(f"  Best mAP50: {self.best_map50:.4f}")
        print("=" * 70 + "\n")

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"


def train():
    # Eski natijalarni tozalash
    runs_dir = TRAINING_DIR / "runs" / "student-behavior"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
        print(f"Eski natijalar o'chirildi: {runs_dir}")

    # YOLOv8s pretrained modelni yuklash
    model = YOLO("yolov8s.pt")

    # Progress callback
    progress = TrainingProgress(TOTAL_EPOCHS)
    model.add_callback("on_train_start", progress.on_train_start)
    model.add_callback("on_train_epoch_start", progress.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", progress.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", progress.on_fit_epoch_end)
    model.add_callback("on_train_end", progress.on_train_end)

    # O'qitish
    results = model.train(
        data=str(DATASET_YAML),
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(TRAINING_DIR / "runs"),
        name="student-behavior",
        exist_ok=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # best.pt ni models/ ga ko'chirish
    best_pt = TRAINING_DIR / "runs" / "student-behavior" / "weights" / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"Model saved to: {dest}")
    else:
        print("WARNING: best.pt not found!")

    return results


if __name__ == "__main__":
    train()
