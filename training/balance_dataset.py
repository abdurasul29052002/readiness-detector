"""
Crop datasetni balanslashtirish (C variant — Hybrid):
  - Majority (focus/write/read) undersample → train 20K / val 2K
  - Minority (turn-head/discuss/hand-raising/standing) bor holicha qoldiriladi
  - teacher klassi drop qilinadi (604 ta juda kam, domain ham boshqa)

Qolgan kichik imbalance (~5x) training vaqtida WeightedRandomSampler
va class weight bilan qoplanadi.
"""

import random
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "crop_dataset"
DST = BASE / "balanced_dataset"

MAJORITY = {"focus", "write", "read"}
DROP = {"teacher"}

CAPS = {
    "train": 20_000,
    "val": 2_000,
}

SEED = 42


def balance_split(split: str):
    cap = CAPS[split]
    src_split = SRC / split
    dst_split = DST / split

    total_orig = 0
    total_out = 0

    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        if cls in DROP:
            print(f"  {cls:15s}: DROPPED")
            continue

        images = list(cls_dir.glob("*.jpg"))
        original = len(images)
        total_orig += original

        if cls in MAJORITY and original > cap:
            random.seed(SEED)
            selected = random.sample(images, cap)
            status = f"-> {cap}"
        else:
            selected = images
            status = "hammasi"

        out_dir = dst_split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in selected:
            shutil.copy2(img, out_dir / img.name)

        total_out += len(selected)
        print(f"  {cls:15s}: {original:>7d} -> {len(selected):>7d}  ({status})")

    print(f"  JAMI: {total_orig:,} -> {total_out:,}")


def main():
    print(f"Input:  {SRC}")
    print(f"Output: {DST}")

    if DST.exists():
        shutil.rmtree(DST)
        print("Eski balanced_dataset o'chirildi")

    for split in ("train", "val"):
        print(f"\n=== {split.upper()} (cap={CAPS[split]}) ===")
        balance_split(split)

    print(f"\nTayyor! Natija: {DST}")


if __name__ == "__main__":
    main()
