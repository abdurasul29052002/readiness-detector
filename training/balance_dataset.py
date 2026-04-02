"""
Crop datasetni balanslashtirish — katta klasslarni undersampling qilish.

Har bir klass uchun maksimal rasm soni belgilanadi.
Ortiqcha rasmlar random tanlanib, faqat tanlanganlari yangi papkaga ko'chiriladi.

Ishlatish:
  python balance_dataset.py
"""

import random
import shutil
from pathlib import Path

# Har bir klass uchun maksimal rasm soni
MAX_PER_CLASS = 100_000

CLASS_NAMES = [
    "hand-raising",
    "read",
    "write",
    "discuss",
    "bow-head",
    "turn-head",
    "standing",
]

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "crop_dataset"
OUTPUT_DIR = BASE_DIR / "balanced_dataset"

SEED = 42


def balance_split(split: str):
    """Bitta split (train/val) ni balanslashtiradi."""
    print(f"\n{'='*50}")
    print(f"  {split.upper()}")
    print(f"{'='*50}")

    total_original = 0
    total_balanced = 0

    for cls_name in CLASS_NAMES:
        src_dir = INPUT_DIR / split / cls_name
        dst_dir = OUTPUT_DIR / split / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Barcha rasmlarni yig'ish
        images = list(src_dir.glob("*.jpg"))
        original_count = len(images)
        total_original += original_count

        # Agar MAX_PER_CLASS dan kam bo'lsa — hammasini olish
        if original_count <= MAX_PER_CLASS:
            selected = images
        else:
            random.seed(SEED)
            selected = random.sample(images, MAX_PER_CLASS)

        # Ko'chirish
        for img_path in selected:
            shutil.copy2(img_path, dst_dir / img_path.name)

        balanced_count = len(selected)
        total_balanced += balanced_count

        # Statistika
        ratio = balanced_count / original_count * 100 if original_count > 0 else 0
        status = "hammasi" if original_count <= MAX_PER_CLASS else f"-> {MAX_PER_CLASS}"
        print(f"  {cls_name:15s}: {original_count:>7d} -> {balanced_count:>7d}  ({status})")

    print(f"\n  Jami: {total_original:,} -> {total_balanced:,}")
    return total_original, total_balanced


def main():
    print(f"Max per class: {MAX_PER_CLASS:,}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Eski natijalarni tozalash
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print("Eski balanced_dataset o'chirildi")

    grand_original = 0
    grand_balanced = 0

    for split in ("train", "val"):
        orig, bal = balance_split(split)
        grand_original += orig
        grand_balanced += bal

    print(f"\n{'='*50}")
    print(f"  TAYYOR!")
    print(f"  {grand_original:,} -> {grand_balanced:,} rasm")
    print(f"  Tejaldi: {grand_original - grand_balanced:,} rasm")
    print(f"  Natija: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
