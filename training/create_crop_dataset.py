import cv2
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
CROP_DIR = ROOT / "crop_dataset"

CLASS_NAMES = {
    0: "write",
    1: "read",
    2: "focus",
    3: "turn-head",
    4: "hand-raising",
    5: "standing",
    6: "discuss",
    7: "teacher",
}

PADDING = 0.1


def crop_bbox(img, cx, cy, w, h, padding=PADDING):
    ih, iw = img.shape[:2]
    x1 = int((cx - w / 2 - padding * w) * iw)
    y1 = int((cy - h / 2 - padding * h) * ih)
    x2 = int((cx + w / 2 + padding * w) * iw)
    y2 = int((cy + h / 2 + padding * h) * ih)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def process_split(split: str):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    out_dir = CROP_DIR / split
    counts = Counter()

    label_files = sorted(lbl_dir.glob("*.txt"))
    total = len(label_files)

    for i, lbl_path in enumerate(label_files):
        img_path = img_dir / lbl_path.with_suffix(".jpg").name
        if not img_path.exists():
            for ext in (".png", ".jpeg", ".bmp"):
                alt = img_dir / lbl_path.with_suffix(ext).name
                if alt.exists():
                    img_path = alt
                    break

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            crop = crop_bbox(img, cx, cy, w, h)
            if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            cls_dir = out_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            idx = counts[f"{split}_{cls_id}"]
            counts[f"{split}_{cls_id}"] += 1
            fname = f"{lbl_path.stem}_{idx}.jpg"
            cv2.imwrite(str(cls_dir / fname), crop)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {split}: {i + 1}/{total} images processed")

    return counts


def main():
    print("Crop dataset yaratilmoqda...")
    print(f"Manba: {DATASET}")
    print(f"Natija: {CROP_DIR}")
    print(f"Padding: {PADDING * 100:.0f}%\n")

    for split in ("train", "val"):
        print(f"[{split}]")
        counts = process_split(split)
        total = 0
        for cls_id, cls_name in sorted(CLASS_NAMES.items()):
            key = f"{split}_{cls_id}"
            n = counts.get(key, 0)
            total += n
            print(f"  {cls_name}: {n}")
        print(f"  JAMI: {total}\n")

    print("Tayyor!")


if __name__ == "__main__":
    main()
