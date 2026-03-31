"""
Kaggle datasetini bizning datasetga qo'shish scripti.
- Kaggle sinf raqamlarini bizning sinflarimizga map qiladi
- Teacher guidance (7) ni o'tkazib yuboradi
- Standing (4) ni yangi sinf 6 sifatida qo'shadi
"""

import os
import shutil
from pathlib import Path

# Paths
KAGGLE_DIR = Path("C:/Users/abdur/Downloads/archive")
OUR_DIR = Path("C:/Users/abdur/IdeaProjects/emotion-detector/training/dataset")

# Kaggle class -> Our class mapping
# Kaggle: 0=write_heads_down, 1=listen, 2=raising_hands, 3=turning_heads,
#         4=standing, 5=group_discussion, 6=looking_at_book, 7=teacher_guidance
CLASS_MAP = {
    "0": "2",   # write heads down -> write
    "1": "1",   # listen/look up -> read
    "2": "0",   # raising hands -> hand-raising
    "3": "5",   # turning heads -> TurnHead
    "4": "6",   # standing -> standing (NEW)
    "5": "3",   # group discussion -> discuss
    "6": "1",   # looking at book -> read
    # "7" -> skip (teacher guidance)
}

stats = {"copied": 0, "skipped_empty": 0, "skipped_teacher_only": 0, "annotations_total": 0, "annotations_skipped": 0}


def remap_labels(src_label_path):
    """Remap class IDs in a label file. Returns remapped lines or None if empty."""
    remapped = []
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = parts[0]
            if cls == "7":
                stats["annotations_skipped"] += 1
                continue
            if cls in CLASS_MAP:
                parts[0] = CLASS_MAP[cls]
                remapped.append(" ".join(parts))
                stats["annotations_total"] += 1
    return remapped


def process_split(split):
    kaggle_imgs = KAGGLE_DIR / "images" / split
    kaggle_labels = KAGGLE_DIR / "labels" / split
    our_imgs = OUR_DIR / "images" / split
    our_labels = OUR_DIR / "labels" / split

    label_files = [f for f in os.listdir(kaggle_labels) if f.endswith(".txt")]
    print(f"\n--- {split.upper()} ---")
    print(f"Kaggle {split} labels: {len(label_files)}")

    for label_file in label_files:
        src_label = kaggle_labels / label_file
        remapped = remap_labels(src_label)

        if not remapped:
            stats["skipped_teacher_only"] += 1
            continue

        # Find corresponding image
        img_stem = Path(label_file).stem
        img_src = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = kaggle_imgs / (img_stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            stats["skipped_empty"] += 1
            continue

        # Copy image
        shutil.copy2(img_src, our_imgs / img_src.name)

        # Write remapped label
        dst_label = our_labels / label_file
        with open(dst_label, "w") as f:
            f.write("\n".join(remapped) + "\n")

        stats["copied"] += 1

    print(f"Copied: {stats['copied']}")


# Delete old cache files
for split in ["train", "val"]:
    cache = OUR_DIR / "labels" / split / f"{split}.cache"
    if cache.exists():
        cache.unlink()
        print(f"Deleted cache: {cache}")

process_split("train")
process_split("val")

print(f"\n=== FINAL STATS ===")
print(f"Images copied: {stats['copied']}")
print(f"Annotations added: {stats['annotations_total']}")
print(f"Teacher annotations skipped: {stats['annotations_skipped']}")
print(f"Images skipped (teacher-only): {stats['skipped_teacher_only']}")
print(f"Images skipped (no image found): {stats['skipped_empty']}")
