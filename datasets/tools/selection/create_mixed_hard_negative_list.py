import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "segmentation"
POSITIVE_FILE = DATASET_ROOT / "train_positive.txt"
HARD_NEGATIVE_FILE = DATASET_ROOT / "mined_hard_negatives.txt"
OUTPUT_FILE = DATASET_ROOT / "train_mixed_hard_negative.txt"
SUMMARY_FILE = DATASET_ROOT / "mixed_hard_negative_summary.json"
SEED = 61


def read_paths(list_path):
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing list: {list_path}")

    raw_lines = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_lines) != len(set(raw_lines)):
        raise RuntimeError(f"Duplicate entries found in {list_path}")

    paths = []
    for line in raw_lines:
        path = Path(line)
        if not path.is_absolute():
            path = DATASET_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        paths.append(path)

    return paths


def label_path_for(image_path):
    try:
        relative = image_path.relative_to(DATASET_ROOT / "images" / "train")
    except ValueError as exc:
        raise RuntimeError(f"Image is outside the training split: {image_path}") from exc
    return DATASET_ROOT / "labels" / "train" / relative.with_suffix(".txt")


def count_objects(image_paths, expect_positive):
    object_count = 0

    for image_path in image_paths:
        label_path = label_path_for(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path}")

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            if line.strip()
        ]

        if expect_positive and not lines:
            raise RuntimeError(f"Positive image has an empty label: {image_path}")
        if not expect_positive and lines:
            raise RuntimeError(f"Negative image has a non-empty label: {image_path}")

        object_count += len(lines)

    return object_count


def main():
    positives = read_paths(POSITIVE_FILE)
    hard_negatives = read_paths(HARD_NEGATIVE_FILE)

    if len(positives) != 570:
        raise RuntimeError(f"Expected 570 positives, found {len(positives)}.")
    if len(hard_negatives) != 570:
        raise RuntimeError(
            f"Expected 570 hard negatives, found {len(hard_negatives)}."
        )

    positive_set = set(positives)
    negative_set = set(hard_negatives)
    overlap = positive_set & negative_set
    if overlap:
        raise RuntimeError(f"Positive/negative overlap found: {sorted(overlap)[:5]}")

    object_count = count_objects(positives, expect_positive=True)
    count_objects(hard_negatives, expect_positive=False)

    combined = positives + hard_negatives
    random.Random(SEED).shuffle(combined)

    OUTPUT_FILE.write_text(
        "\n".join(str(path) for path in combined) + "\n",
        encoding="utf-8",
    )

    summary = {
        "positive_images": len(positives),
        "hard_negative_images": len(hard_negatives),
        "total_training_images": len(combined),
        "positive_objects": object_count,
        "positive_to_negative_ratio": "1:1",
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
