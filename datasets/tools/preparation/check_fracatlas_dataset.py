from collections import Counter
from pathlib import Path

DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fracatlas_yolo"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_split(split):
    """Check image and label consistency for one dataset split.

    Args:
        split: Dataset split name.

    Returns:
        True if image and label files match by stem.
    """
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    labels = list(label_dir.glob("*.txt"))

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    positive_images = 0
    negative_images = 0
    class_counts = Counter()

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        content = label.read_text(encoding="utf-8", errors="ignore").strip()

        if not content:
            negative_images += 1
            continue

        positive_images += 1

        for line in content.splitlines():
            parts = line.split()

            if parts:
                class_counts[int(float(parts[0]))] += 1

    print(f"{split}:")
    print(f"  Image count: {len(images)}")
    print(f"  Positive images: {positive_images}")
    print(f"  Negative images: {negative_images}")
    print(f"  Missing labels: {len(missing_labels)}")
    print(f"  Orphan labels: {len(orphan_labels)}")
    print(f"  Objects by class: {dict(sorted(class_counts.items()))}")

    if missing_labels:
        print(f"  Missing label examples: {missing_labels[:5]}")

    if orphan_labels:
        print(f"  Orphan label examples: {orphan_labels[:5]}")

    return len(missing_labels) == 0 and len(orphan_labels) == 0


if __name__ == "__main__":
    passed = True

    for split_name in ("train", "val", "test"):
        if not check_split(split_name):
            passed = False

    if passed:
        print("Dataset structure check passed, no source files were modified.")
    else:
        raise SystemExit("Dataset structure check failed, fix the dataset before training.")
