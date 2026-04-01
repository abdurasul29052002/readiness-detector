"""
Mavjud YOLO detection datasetdan classification dataset yaratish.

Har bir rasmdan bounding boxlar bo'yicha o'quvchilarni kesib olib,
class bo'yicha papkalarga saqlaydi (ImageNet-style).

Natija: dataset/crop_dataset/{train,val}/{class_name}/*.jpg
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

CLASS_NAMES = {
    0: "hand-raising",
    1: "read",
    2: "write",
    3: "discuss",
    4: "bow-head",
    5: "turn-head",
    6: "standing",
}

# Crop atrofiga qo'shimcha joy (20%)
PADDING_RATIO = 0.2
# Minimal crop o'lchami (piksel) — juda kichik croplarni o'tkazib yuborish
MIN_CROP_SIZE = 20

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "crop_dataset"


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO formatdagi label faylni o'qiydi."""
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def crop_and_save(img, cls_id: int, cx: float, cy: float, bw: float, bh: float,
                  img_h: int, img_w: int, output_path: Path):
    """Rasmdan bitta o'quvchini kesib oladi va saqlaydi."""
    # Padding qo'shish
    pad_w = bw * PADDING_RATIO
    pad_h = bh * PADDING_RATIO

    # Normalized koordinatalarni pikselga aylantirish
    x1 = int((cx - bw / 2 - pad_w) * img_w)
    y1 = int((cy - bh / 2 - pad_h) * img_h)
    x2 = int((cx + bw / 2 + pad_w) * img_w)
    y2 = int((cy + bh / 2 + pad_h) * img_h)

    # Chegaralarni tekshirish
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return False

    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), crop)
    return True


def process_image(args: tuple) -> tuple[int, int]:
    """Bitta rasmni qayta ishlaydi, croplarni saqlaydi."""
    img_path, label_path, split = args
    saved = 0
    skipped = 0

    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0

    img_h, img_w = img.shape[:2]
    boxes = parse_label_file(label_path)

    for i, (cls_id, cx, cy, bw, bh) in enumerate(boxes):
        cls_name = CLASS_NAMES.get(cls_id)
        if cls_name is None:
            skipped += 1
            continue

        out_dir = OUTPUT_DIR / split / cls_name
        out_path = out_dir / f"{img_path.stem}_{i}.jpg"

        if crop_and_save(img, cls_id, cx, cy, bw, bh, img_h, img_w, out_path):
            saved += 1
        else:
            skipped += 1

    return saved, skipped


def collect_tasks(split: str) -> list[tuple]:
    """Train yoki val uchun barcha rasm-label juftliklarini yig'adi."""
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split
    tasks = []

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            tasks.append((img_path, label_path, split))

    return tasks


def main():
    # Papkalar yaratish
    for split in ("train", "val"):
        for cls_name in CLASS_NAMES.values():
            (OUTPUT_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    # Barcha vazifalarni yig'ish
    tasks = []
    for split in ("train", "val"):
        split_tasks = collect_tasks(split)
        tasks.extend(split_tasks)
        print(f"{split}: {len(split_tasks)} ta rasm topildi")

    print(f"\nJami: {len(tasks)} ta rasm qayta ishlanadi...")
    print(f"Natija: {OUTPUT_DIR}\n")

    total_saved = 0
    total_skipped = 0
    done = 0

    workers = 6
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_image, t): t for t in tasks}

        for future in as_completed(futures):
            saved, skipped = future.result()
            total_saved += saved
            total_skipped += skipped
            done += 1

            if done % 2000 == 0 or done == len(tasks):
                pct = done / len(tasks) * 100
                print(f"  [{pct:5.1f}%] {done}/{len(tasks)} rasm — "
                      f"{total_saved} crop saqlandi, {total_skipped} o'tkazildi")

    # Statistika
    print(f"\n{'='*50}")
    print(f"TAYYOR! Jami: {total_saved} crop saqlandi, {total_skipped} o'tkazildi\n")

    for split in ("train", "val"):
        print(f"  {split}:")
        for cls_name in CLASS_NAMES.values():
            cls_dir = OUTPUT_DIR / split / cls_name
            count = len(list(cls_dir.glob("*.jpg")))
            print(f"    {cls_name:15s}: {count:>6d}")
        print()


if __name__ == "__main__":
    main()
