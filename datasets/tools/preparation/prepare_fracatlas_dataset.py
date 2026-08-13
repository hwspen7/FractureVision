from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parents[2]
SOURCE = DATASETS_DIR / "fracatlas_raw" / "FracAtlas"
OUTPUT = DATASETS_DIR / "fracatlas_yolo"
BUILDING = DATASETS_DIR / "fracatlas_yolo_building"

SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.12,
    "test": 0.08,
}

REGION_COLUMNS = ("hand", "leg", "hip", "shoulder", "mixed")
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def read_label(path: Path, is_positive: bool) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing label: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if is_positive and not lines:
        raise ValueError(f"Positive image has empty label: {path}")

    if not is_positive and lines:
        raise ValueError(f"Negative image has non-empty label: {path}")

    for line_number, line in enumerate(lines, 1):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid label row: {path}:{line_number}: {line}"
            )

        class_id = parts[0]
        if class_id != "0":
            raise ValueError(
                f"Unexpected class {class_id}: {path}:{line_number}"
            )

        try:
            x, y, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(
                f"Non-numeric label: {path}:{line_number}"
            ) from error

        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(
                f"Invalid center coordinate: {path}:{line_number}"
            )

        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(
                f"Invalid box size: {path}:{line_number}"
            )

    return lines


def region_name(row: dict[str, str]) -> str:
    regions = [
        column
        for column in REGION_COLUMNS
        if row.get(column) == "1"
    ]
    return "+".join(regions) if regions else "unknown"


def deterministic_order(record: dict) -> str:
    value = f"{SEED}:{record['image_id']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def allocate_negative_split(records: list[dict]) -> None:
    records.sort(key=deterministic_order)
    total = len(records)

    train_count = int(total * SPLIT_RATIOS["train"])
    val_count = int(total * SPLIT_RATIOS["val"])

    for index, record in enumerate(records):
        if index < train_count:
            record["split"] = "train"
        elif index < train_count + val_count:
            record["split"] = "val"
        else:
            record["split"] = "test"


def load_official_positive_splits() -> dict[str, str]:
    split_root = SOURCE / "Utilities" / "Fracture Split"
    mapping: dict[str, str] = {}

    files = {
        "train": "train.csv",
        "val": "valid.csv",
        "test": "test.csv",
    }

    for split, filename in files.items():
        for row in read_csv(split_root / filename):
            image_id = row["image_id"]

            if image_id in mapping:
                raise ValueError(
                    f"Positive image appears in multiple splits: {image_id}"
                )

            mapping[image_id] = split

    return mapping


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "It was not overwritten."
        )

    if BUILDING.exists():
        raise FileExistsError(
            f"Incomplete build directory exists: {BUILDING}\n"
            "Inspect it before trying again."
        )

    dataset_rows = read_csv(SOURCE / "dataset.csv")
    positive_splits = load_official_positive_splits()

    records: list[dict] = []
    csv_ids: set[str] = set()

    for row in dataset_rows:
        image_id = row["image_id"]

        if image_id in csv_ids:
            raise ValueError(f"Duplicate dataset.csv ID: {image_id}")
        csv_ids.add(image_id)

        is_positive = row["fractured"] == "1"
        category = "Fractured" if is_positive else "Non_fractured"

        image_path = SOURCE / "images" / category / image_id
        label_path = (
            SOURCE / "Annotations" / "YOLO"
            / f"{Path(image_id).stem}.txt"
        )

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        label_lines = read_label(label_path, is_positive)

        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "label_path": label_path,
                "label_lines": label_lines,
                "positive": is_positive,
                "region": region_name(row),
                "sha256": calculate_sha256(image_path),
            }
        )

    positive_ids = {
        record["image_id"]
        for record in records
        if record["positive"]
    }

    if positive_ids != set(positive_splits):
        missing = sorted(positive_ids - set(positive_splits))
        extra = sorted(set(positive_splits) - positive_ids)
        raise ValueError(
            f"Official positive split mismatch.\n"
            f"Missing: {missing}\nExtra: {extra}"
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["sha256"]].append(record)

    kept_records: list[dict] = []
    removed_duplicates: list[dict] = []
    label_conflict_groups = 0

    for digest, group in sorted(groups.items()):
        statuses = {record["positive"] for record in group}

        if len(statuses) != 1:
            raise ValueError(
                "Identical image content has conflicting positive/negative "
                f"status: {[r['image_id'] for r in group]}"
            )

        canonical = min(group, key=lambda record: record["image_id"])

        if canonical["positive"]:
            group_splits = {
                positive_splits[record["image_id"]]
                for record in group
            }

            if len(group_splits) != 1:
                raise ValueError(
                    "Duplicate positive image crosses official splits: "
                    f"{[r['image_id'] for r in group]}"
                )

            canonical["split"] = next(iter(group_splits))

            label_versions = {
                tuple(record["label_lines"])
                for record in group
            }
            if len(label_versions) > 1:
                label_conflict_groups += 1

        kept_records.append(canonical)

        for removed in group:
            if removed is canonical:
                continue

            removed_duplicates.append(
                {
                    "kept_id": canonical["image_id"],
                    "removed_id": removed["image_id"],
                    "sha256": digest,
                    "positive": int(canonical["positive"]),
                    "label_different": int(
                        removed["label_lines"]
                        != canonical["label_lines"]
                    ),
                }
            )

    negative_by_region: dict[str, list[dict]] = defaultdict(list)

    for record in kept_records:
        if not record["positive"]:
            negative_by_region[record["region"]].append(record)

    for region_records in negative_by_region.values():
        allocate_negative_split(region_records)

    if any("split" not in record for record in kept_records):
        raise RuntimeError("Some records were not assigned to a split.")

    BUILDING.mkdir(parents=True)

    for split in SPLIT_RATIOS:
        (BUILDING / "images" / split).mkdir(parents=True)
        (BUILDING / "labels" / split).mkdir(parents=True)

    manifest_rows = []

    for record in sorted(
        kept_records,
        key=lambda item: (item["split"], item["image_id"]),
    ):
        split = record["split"]

        target_image = BUILDING / "images" / split / record["image_id"]
        target_label = (
            BUILDING / "labels" / split
            / f"{Path(record['image_id']).stem}.txt"
        )

        shutil.copy2(record["image_path"], target_image)
        shutil.copy2(record["label_path"], target_label)

        manifest_rows.append(
            {
                "image_id": record["image_id"],
                "split": split,
                "fractured": int(record["positive"]),
                "region": record["region"],
                "objects": len(record["label_lines"]),
                "sha256": record["sha256"],
            }
        )

    with (BUILDING / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "split",
                "fractured",
                "region",
                "objects",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (BUILDING / "removed_duplicates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "kept_id",
                "removed_id",
                "sha256",
                "positive",
                "label_different",
            ],
        )
        writer.writeheader()
        writer.writerows(removed_duplicates)

    selected_paths = {
        record["image_path"].resolve()
        for record in records
    }

    all_image_files = sorted((SOURCE / "images").glob("*/*.jpg"))
    ignored_folder_copies = [
        str(path.relative_to(SOURCE))
        for path in all_image_files
        if path.resolve() not in selected_paths
    ]

    split_summary = {}

    for split in SPLIT_RATIOS:
        subset = [
            record for record in kept_records
            if record["split"] == split
        ]

        split_summary[split] = {
            "images": len(subset),
            "positive": sum(r["positive"] for r in subset),
            "negative": sum(not r["positive"] for r in subset),
            "objects": sum(len(r["label_lines"]) for r in subset),
            "regions": dict(
                Counter(r["region"] for r in subset)
            ),
        }

    summary = {
        "source_csv_records": len(records),
        "source_image_files": len(all_image_files),
        "unique_content_images": len(kept_records),
        "removed_duplicate_ids": len(removed_duplicates),
        "duplicate_label_conflict_groups": label_conflict_groups,
        "ignored_wrong_folder_copies": ignored_folder_copies,
        "class_names": {"0": "fracture"},
        "split_ratios_for_negatives": SPLIT_RATIOS,
        "splits": split_summary,
    }

    (BUILDING / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    BUILDING.rename(OUTPUT)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nNew dataset generated: {OUTPUT}")
    print("Original FracAtlas files were not modified.")


if __name__ == "__main__":
    main()