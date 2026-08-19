import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parents[3]
DATASET_ROOT = DATASETS_DIR / "detection"

MANIFEST_FILE = DATASET_ROOT / "split_manifest.csv"
HARD_FILE = DATASET_ROOT / "deprecated" / "mined_hard_negatives.txt"

TRAIN_FILE = DATASET_ROOT / "deprecated" / "train_stage2.txt"
RANDOM_FILE = DATASET_ROOT / "deprecated" / "stage2_random_negatives.txt"
SUMMARY_FILE = DATASET_ROOT / "deprecated" / "stage2_selection_summary.json"

HARD_NEGATIVE_COUNT = 285
RANDOM_NEGATIVE_COUNT = 285
SEED = 43


def deterministic_key(image_id):
    value = f"{SEED}:{image_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def image_line(row):
    return f"./images/train/{row['image_id']}"


def read_manifest():
    with MANIFEST_FILE.open(
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def calculate_quotas(rows, total):
    region_counts = Counter(row["region"] for row in rows)
    population = sum(region_counts.values())

    exact = {
        region: total * count / population
        for region, count in region_counts.items()
    }

    quotas = {
        region: math.floor(value)
        for region, value in exact.items()
    }

    remaining = total - sum(quotas.values())

    regions_by_remainder = sorted(
        exact,
        key=lambda region: (
            -(exact[region] - quotas[region]),
            region,
        ),
    )

    for region in regions_by_remainder[:remaining]:
        quotas[region] += 1

    return quotas


def validate_rows(rows):
    for row in rows:
        image_path = (
            DATASET_ROOT
            / "images"
            / "train"
            / row["image_id"]
        )
        label_path = (
            DATASET_ROOT
            / "labels"
            / "train"
            / f"{Path(row['image_id']).stem}.txt"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Training image not found: {image_path}"
            )

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Training label not found: {label_path}"
            )

        label_content = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if row["fractured"] == "1" and not label_content:
            raise ValueError(
                f"Positive image has an empty label: {row['image_id']}"
            )

        if row["fractured"] == "0" and label_content:
            raise ValueError(
                f"Negative image has a non-empty label: {row['image_id']}"
            )


def main():
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(
            f"Manifest file not found: {MANIFEST_FILE}"
        )

    if not HARD_FILE.is_file():
        raise FileNotFoundError(
            f"Hard-negative list not found: {HARD_FILE}"
        )

    rows = read_manifest()

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

    row_by_id = {
        row["image_id"]: row
        for row in train_rows
    }

    hard_lines = [
        line.strip()
        for line in HARD_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(hard_lines) != len(set(hard_lines)):
        raise ValueError("Hard-negative list contains duplicate images.")

    hard_ids = [
        Path(line).name
        for line in hard_lines[:HARD_NEGATIVE_COUNT]
    ]

    hard_rows = []

    for image_id in hard_ids:
        if image_id not in row_by_id:
            raise ValueError(
                f"Hard-negative image is not in the training set: {image_id}"
            )

        row = row_by_id[image_id]

        if row["fractured"] != "0":
            raise ValueError(
                f"Hard-negative image is not actually negative: {image_id}"
            )

        hard_rows.append(row)

    if len(hard_rows) != HARD_NEGATIVE_COUNT:
        raise ValueError(
            f"Expected {HARD_NEGATIVE_COUNT} hard negatives, "
            f"got {len(hard_rows)}."
        )

    hard_id_set = set(hard_ids)

    random_candidates = [
        row for row in negatives
        if row["image_id"] not in hard_id_set
    ]

    quotas = calculate_quotas(
        random_candidates,
        RANDOM_NEGATIVE_COUNT,
    )

    rows_by_region = defaultdict(list)

    for row in random_candidates:
        rows_by_region[row["region"]].append(row)

    random_rows = []

    for region, quota in sorted(quotas.items()):
        region_rows = sorted(
            rows_by_region[region],
            key=lambda row: deterministic_key(
                row["image_id"]
            ),
        )

        random_rows.extend(region_rows[:quota])

    if len(random_rows) != RANDOM_NEGATIVE_COUNT:
        raise RuntimeError(
            f"Expected {RANDOM_NEGATIVE_COUNT} random negatives, "
            f"got {len(random_rows)}."
        )

    selected_rows = positives + hard_rows + random_rows

    selected_ids = [
        row["image_id"]
        for row in selected_rows
    ]

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Stage 2 training list contains duplicate images.")

    validate_rows(selected_rows)

    selected_rows.sort(
        key=lambda row: deterministic_key(
            row["image_id"]
        )
    )

    TRAIN_FILE.write_text(
        "\n".join(
            image_line(row)
            for row in selected_rows
        ) + "\n",
        encoding="utf-8",
    )

    RANDOM_FILE.write_text(
        "\n".join(
            image_line(row)
            for row in random_rows
        ) + "\n",
        encoding="utf-8",
    )

    summary = {
        "positive_images": len(positives),
        "hard_negative_images": len(hard_rows),
        "random_negative_images": len(random_rows),
        "total_negative_images": (
            len(hard_rows) + len(random_rows)
        ),
        "total_training_images": len(selected_rows),
        "positive_to_negative_ratio": "1:1",
        "hard_negative_regions": dict(
            sorted(
                Counter(
                    row["region"]
                    for row in hard_rows
                ).items()
            )
        ),
        "random_negative_regions": dict(
            sorted(
                Counter(
                    row["region"]
                    for row in random_rows
                ).items()
            )
        ),
        "seed": SEED,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"Stage 2 training list: {TRAIN_FILE}")
    print(f"Random-negative list: {RANDOM_FILE}")
    print("No images, labels, validation data, or test data were modified.")


if __name__ == "__main__":
    main()
