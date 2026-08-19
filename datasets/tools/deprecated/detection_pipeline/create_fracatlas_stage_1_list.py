import csv
import hashlib
import json
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parents[3]
DATASET_ROOT = DATASETS_DIR / "detection"
MANIFEST = DATASET_ROOT / "split_manifest.csv"

TRAIN_LIST = DATASET_ROOT / "deprecated" / "train_selected.txt"
HARD_NEGATIVE_POOL = DATASET_ROOT / "deprecated" / "hard_negative_pool.txt"
SUMMARY_FILE = DATASET_ROOT / "deprecated" / "train_selection_summary.json"

NEGATIVES_PER_POSITIVE = 2
SEED = 42


def deterministic_key(row: dict[str, str]) -> str:
    value = f"{SEED}:{row['image_id']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def image_line(row: dict[str, str]) -> str:
    return f"./images/train/{row['image_id']}"


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    positives = [
        row for row in train_rows
        if row["fractured"] == "1"
    ]

    negatives = [
        row for row in train_rows
        if row["fractured"] == "0"
    ]

    positives.sort(key=deterministic_key)
    negatives.sort(key=deterministic_key)

    negative_limit = min(
        len(negatives),
        len(positives) * NEGATIVES_PER_POSITIVE,
    )

    selected_negatives = negatives[:negative_limit]
    unused_negatives = negatives[negative_limit:]

    selected_rows = positives + selected_negatives
    selected_rows.sort(key=deterministic_key)

    TRAIN_LIST.write_text(
        "\n".join(image_line(row) for row in selected_rows) + "\n",
        encoding="utf-8",
    )

    HARD_NEGATIVE_POOL.write_text(
        "\n".join(image_line(row) for row in unused_negatives) + "\n",
        encoding="utf-8",
    )

    summary = {
        "positive_images": len(positives),
        "selected_negative_images": len(selected_negatives),
        "selected_training_images": len(selected_rows),
        "reserved_hard_negative_images": len(unused_negatives),
        "negative_to_positive_ratio": NEGATIVES_PER_POSITIVE,
        "seed": SEED,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nTraining list: {TRAIN_LIST}")
    print(f"Hard-negative pool: {HARD_NEGATIVE_POOL}")
    print("No images or labels were modified.")


if __name__ == "__main__":
    main()