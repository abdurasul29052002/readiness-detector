"""
3 ta datasetni ZIP dan ochib, class ID larni remap qilib,
training/dataset/ papkasiga birlashtiradi.

Class mapping:
  Handrise-Read-write: 0->0 (hand-raising), 1->1 (read), 2->2 (write)
  Discuss:             0->3 (discuss)
  BowTurnHead:         0->4 (BowHead), 1->5 (TurnHead)
"""

import os
import zipfile
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SCB_DIR = BASE_DIR / "SCB-Dataset"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"

ZIPS = [
    {
        "zip": SCB_DIR / "SCB5-Handrise-Read-write-2024-9-17" / "SCB5-Handrise-Read-write-2024-9-17.zip",
        "inner_root": "SCB5-Handrise-Read-write-2024-9-17",
        "class_remap": {0: 0, 1: 1, 2: 2},
        "prefix": "hrw",
    },
    {
        "zip": SCB_DIR / "SCB5-Discuss-2024-9-17" / "SCB5-Discuss-2024-9-17.zip",
        "inner_root": "SCB5-Discuss-2024-9-17",
        "class_remap": {0: 3},
        "prefix": "disc",
    },
    {
        "zip": SCB_DIR / "SCB_BowTurnHead_20250509" / "SCB_BowTurnHead_20250509.zip",
        "inner_root": "SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17",
        "class_remap": {0: 4, 1: 5},
        "prefix": "bth",
    },
]


def prepare_dirs():
    """Dataset papkalarini yaratish."""
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def remap_label(content: str, class_remap: dict[int, int]) -> str:
    """Label fayldagi class ID larni remap qilish."""
    lines = []
    for line in content.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_id = int(parts[0])
        new_id = class_remap.get(old_id)
        if new_id is None:
            print(f"  WARNING: unknown class {old_id}, skipping line")
            continue
        parts[0] = str(new_id)
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n" if lines else ""


def process_zip(zip_info: dict):
    """Bitta ZIP ni ochib, dataset papkasiga ko'chirish."""
    zip_path = zip_info["zip"]
    inner_root = zip_info["inner_root"]
    class_remap = zip_info["class_remap"]
    prefix = zip_info["prefix"]

    print(f"\n{'='*60}")
    print(f"Processing: {zip_path.name}")
    print(f"  Inner root: {inner_root}")
    print(f"  Remap: {class_remap}")

    counts = {"train": 0, "val": 0}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ("train", "val"):
            img_prefix = f"{inner_root}/images/{split}/"
            lbl_prefix = f"{inner_root}/labels/{split}/"

            # Collect image files
            img_files = [
                n for n in zf.namelist()
                if n.startswith(img_prefix) and not n.endswith("/")
            ]

            for img_path in img_files:
                filename = os.path.basename(img_path)
                name_stem = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]

                # New filenames with prefix to avoid collisions
                new_img_name = f"{prefix}_{filename}"
                new_lbl_name = f"{prefix}_{name_stem}.txt"

                # Extract and copy image
                img_data = zf.read(img_path)
                img_dest = DATASET_DIR / "images" / split / new_img_name
                img_dest.write_bytes(img_data)

                # Extract, remap and copy label
                lbl_path = f"{lbl_prefix}{name_stem}.txt"
                try:
                    lbl_data = zf.read(lbl_path).decode("utf-8")
                    remapped = remap_label(lbl_data, class_remap)
                    if remapped:
                        lbl_dest = DATASET_DIR / "labels" / split / new_lbl_name
                        lbl_dest.write_text(remapped, encoding="utf-8")
                        counts[split] += 1
                except KeyError:
                    print(f"  WARNING: no label for {img_path}")

    print(f"  Done: train={counts['train']}, val={counts['val']}")
    return counts


def verify_dataset():
    """Dataset ni tekshirish."""
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    for split in ("train", "val"):
        imgs = list((DATASET_DIR / "images" / split).glob("*"))
        lbls = list((DATASET_DIR / "labels" / split).glob("*"))
        print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")

    # Count classes
    print("\nClass distribution:")
    class_counts = {}
    for split in ("train", "val"):
        for lbl_file in (DATASET_DIR / "labels" / split).glob("*.txt"):
            for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
                cls = int(line.split()[0])
                key = (split, cls)
                class_counts[key] = class_counts.get(key, 0) + 1

    names = {0: "hand-raising", 1: "read", 2: "write", 3: "discuss", 4: "BowHead", 5: "TurnHead"}
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'-'*41}")
    total_t, total_v = 0, 0
    for cls_id in range(6):
        t = class_counts.get(("train", cls_id), 0)
        v = class_counts.get(("val", cls_id), 0)
        total_t += t
        total_v += v
        print(f"  {cls_id} {names[cls_id]:<13} {t:>8} {v:>8} {t+v:>8}")
    print(f"  {'-'*41}")
    print(f"  {'TOTAL':<15} {total_t:>8} {total_v:>8} {total_t+total_v:>8}")


if __name__ == "__main__":
    print("Dataset preparation started...")

    if DATASET_DIR.exists():
        print(f"Removing old dataset dir: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    prepare_dirs()

    total = {"train": 0, "val": 0}
    for z in ZIPS:
        counts = process_zip(z)
        total["train"] += counts["train"]
        total["val"] += counts["val"]

    print(f"\n{'='*60}")
    print(f"TOTAL: train={total['train']}, val={total['val']}, all={total['train']+total['val']}")

    verify_dataset()
    print("\nDone!")
