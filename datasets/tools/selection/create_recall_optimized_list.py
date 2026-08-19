import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "segmentation"
IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"

POSITIVE_FILE = DATASET_ROOT / "train_positive.txt"
MINED_HARD_FILE = DATASET_ROOT / "mined_hard_negatives.txt"
OUTPUT_FILE = DATASET_ROOT / "train_recall_optimized.txt"
RANDOM_NEGATIVE_FILE = DATASET_ROOT / "recall_random_negatives.txt"
HARD_NEGATIVE_FILE = DATASET_ROOT / "recall_hard_negatives.txt"
SUMMARY_FILE = DATASET_ROOT / "recall_optimized_summary.json"

HARD_NEGATIVE_COUNT = 143
RANDOM_NEGATIVE_COUNT = 142
SEED = 81
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_list_line(line):
    path = Path(line)
    if not path.is_absolute():
        path = DATASET_ROOT / path
    return path.resolve()


def read_list(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing list: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"Duplicate entries found in {path}")

    resolved = [resolve_list_line(line) for line in lines]
    missing = [image_path for image_path in resolved if not image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images referenced by {path}: {missing[:5]}")
    return resolved


def label_path_for(image_path):
    try:
        relative = image_path.relative_to(IMAGE_DIR.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Image is outside the training split: {image_path}") from exc
    return LABEL_DIR / relative.with_suffix(".txt")


def label_rows(image_path):
    label_path = label_path_for(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    return [
        row.strip()
        for row in label_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        if row.strip()
    ]


def main():
    positives = read_list(POSITIVE_FILE)
    mined_hard_negatives = read_list(MINED_HARD_FILE)

    if len(positives) != 570:
        raise RuntimeError(f"Expected 570 positives, found {len(positives)}.")
    if len(mined_hard_negatives) != 570:
        raise RuntimeError(
            f"Expected 570 mined hard negatives, found {len(mined_hard_negatives)}."
        )

    positive_objects = 0
    for image_path in positives:
        rows = label_rows(image_path)
        if not rows:
            raise RuntimeError(f"Positive image has an empty label: {image_path}")
        positive_objects += len(rows)

    for image_path in mined_hard_negatives:
        if label_rows(image_path):
            raise RuntimeError(f"Hard negative has a non-empty label: {image_path}")

    if positive_objects != 757:
        raise RuntimeError(f"Expected 757 positive objects, found {positive_objects}.")

    selected_hard = mined_hard_negatives[:HARD_NEGATIVE_COUNT]
    excluded_hard = set(mined_hard_negatives)

    all_negative_images = []
    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_path = image_path.resolve()
        if not label_rows(image_path):
            all_negative_images.append(image_path)

    if len(all_negative_images) != 2678:
        raise RuntimeError(
            f"Expected 2678 training negatives, found {len(all_negative_images)}."
        )

    random_pool = [
        image_path
        for image_path in all_negative_images
        if image_path not in excluded_hard
    ]
    if len(random_pool) < RANDOM_NEGATIVE_COUNT:
        raise RuntimeError("Not enough unused random negatives.")

    rng = random.Random(SEED)
    selected_random = rng.sample(random_pool, RANDOM_NEGATIVE_COUNT)

    combined = positives + selected_hard + selected_random
    rng.shuffle(combined)

    if len(combined) != 855:
        raise RuntimeError(f"Expected 855 training images, found {len(combined)}.")
    if len(combined) != len(set(combined)):
        raise RuntimeError("Duplicate paths found in the recall training list.")

    HARD_NEGATIVE_FILE.write_text(
        "\n".join(str(path) for path in selected_hard) + "\n",
        encoding="utf-8",
    )
    RANDOM_NEGATIVE_FILE.write_text(
        "\n".join(str(path) for path in selected_random) + "\n",
        encoding="utf-8",
    )
    OUTPUT_FILE.write_text(
        "\n".join(str(path) for path in combined) + "\n",
        encoding="utf-8",
    )

    summary = {
        "positive_images": len(positives),
        "hard_negative_images": len(selected_hard),
        "random_negative_images": len(selected_random),
        "total_negative_images": len(selected_hard) + len(selected_random),
        "total_training_images": len(combined),
        "positive_objects": positive_objects,
        "positive_to_negative_ratio": "2:1",
        "hard_to_random_negative_ratio": "approximately 1:1",
        "duplicates": len(combined) - len(set(combined)),
        "seed": SEED,
        "output": str(OUTPUT_FILE),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("No images, labels, validation data, test data, or weights were modified.")


if __name__ == "__main__":
    main()
