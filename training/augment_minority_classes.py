"""
Kam sinflar uchun oversampling + augmentation.
Har bir kam sinf rasmini augmentatsiya qilib ko'paytiradi.
Maqsad: har bir sinf kamida TARGET_MIN annotatsiyaga yetishi.
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

try:
    import cv2
    import numpy as np
except ImportError:
    print("Installing dependencies...")
    os.system("pip install opencv-python numpy")
    import cv2
    import numpy as np

DATASET_DIR = Path("C:/Users/abdur/IdeaProjects/emotion-detector/training/dataset")
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"

TARGET_MIN = 20000  # Har bir sinf kamida shu annotatsiyaga yetsin


def get_class_stats():
    """Har bir sinf uchun annotatsiya soni va qaysi rasmlarda borligini hisoblash."""
    class_counts = Counter()
    class_to_files = defaultdict(list)  # class_id -> [label_file, ...]

    for label_file in os.listdir(TRAIN_LABELS):
        if not label_file.endswith(".txt"):
            continue
        filepath = TRAIN_LABELS / label_file
        classes_in_file = set()
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = parts[0]
                    class_counts[cls] += 1
                    classes_in_file.add(cls)
        for cls in classes_in_file:
            class_to_files[cls].append(label_file)

    return class_counts, class_to_files


def augment_image(img):
    """Rasmga random augmentatsiya qo'llash."""
    augmented = img.copy()
    h, w = augmented.shape[:2]

    # Random horizontal flip (50%)
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Random brightness shift
    brightness = random.randint(-40, 40)
    augmented = np.clip(augmented.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    # Random contrast
    contrast = random.uniform(0.7, 1.3)
    mean = np.mean(augmented)
    augmented = np.clip((augmented.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # Random slight rotation (-10 to +10 degrees)
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random HSV hue shift
    if random.random() > 0.5:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(-10, 10)) % 180
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return augmented


def flip_labels_horizontal(lines):
    """Horizontal flip uchun label koordinatalarini o'zgartirish."""
    flipped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[1] = str(1.0 - float(parts[1]))  # x_center ni flip
            flipped.append(" ".join(parts))
        else:
            flipped.append(line.strip())
    return flipped


def augment_labels(lines, was_flipped):
    """Agar flip bo'lsa labellarni moslashtirish."""
    if was_flipped:
        return flip_labels_horizontal(lines)
    return [l.strip() for l in lines]


def main():
    print("Sinf statistikasini hisoblamoqda...")
    class_counts, class_to_files = get_class_stats()

    print("\nJoriy holat:")
    names = {
        "0": "hand-raising", "1": "read", "2": "write",
        "3": "discuss", "4": "bow-head", "5": "turn-head", "6": "standing"
    }
    for cls in sorted(class_counts.keys(), key=int):
        print(f"  {cls} ({names.get(cls, '?')}): {class_counts[cls]} annotations, {len(class_to_files[cls])} images")

    # Qaysi sinflar kam?
    minority_classes = {cls: count for cls, count in class_counts.items() if count < TARGET_MIN}
    if not minority_classes:
        print(f"\nBarcha sinflar {TARGET_MIN} dan ko'p. Augmentation kerak emas.")
        return

    print(f"\nKam sinflar (< {TARGET_MIN}):")
    for cls, count in sorted(minority_classes.items(), key=lambda x: x[1]):
        needed = TARGET_MIN - count
        print(f"  {cls} ({names.get(cls, '?')}): {count} bor, ~{needed} kerak")

    total_created = 0

    for cls in sorted(minority_classes.keys(), key=int):
        current = class_counts[cls]
        needed = TARGET_MIN - current
        source_files = class_to_files[cls]

        if not source_files:
            continue

        # Har bir source rasmdan nechta nusxa kerak
        copies_per_image = max(1, needed // len(source_files)) + 1
        created = 0

        print(f"\nClass {cls} ({names.get(cls, '?')}): {len(source_files)} rasmdan {copies_per_image}x augment...")

        random.shuffle(source_files)

        for label_file in source_files:
            if created >= needed:
                break

            stem = Path(label_file).stem

            # Find image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = TRAIN_IMAGES / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # Read image and labels
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            with open(TRAIN_LABELS / label_file) as f:
                label_lines = f.readlines()

            for i in range(copies_per_image):
                if created >= needed:
                    break

                # Augment
                aug_img = augment_image(img)
                was_flipped = random.random() > 0.5
                if was_flipped:
                    aug_img = cv2.flip(aug_img, 1)

                aug_labels = augment_labels(label_lines, was_flipped)

                # Save
                new_name = f"aug_{cls}_{stem}_{i}"
                cv2.imwrite(str(TRAIN_IMAGES / f"{new_name}.jpg"), aug_img)
                with open(TRAIN_LABELS / f"{new_name}.txt", "w") as f:
                    f.write("\n".join(aug_labels) + "\n")

                created += 1

        total_created += created
        print(f"  Yaratildi: {created} ta yangi rasm")

    # Final stats
    print(f"\n=== YAKUNIY ===")
    print(f"Jami yangi rasmlar: {total_created}")
    print(f"\nYangi statistika:")
    class_counts2, _ = get_class_stats()
    for cls in sorted(class_counts2.keys(), key=int):
        old = class_counts.get(cls, 0)
        new = class_counts2[cls]
        diff = new - old
        marker = f" (+{diff})" if diff > 0 else ""
        print(f"  {cls} ({names.get(cls, '?')}): {new}{marker}")


if __name__ == "__main__":
    main()
