import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "datasets" / "detection"

DEFAULT_MODEL = (
    ROOT
    / "experiments"
    / "detection"
    / "training"
    / "baseline"
    / "weights"
    / "best.pt"
)

POOL_FILE = DATASET_ROOT / "deprecated" / "hard_negative_pool.txt"
SCORES_FILE = DATASET_ROOT / "deprecated" / "hard_negative_scores.csv"
SELECTED_FILE = DATASET_ROOT / "deprecated" / "mined_hard_negatives.txt"
SUMMARY_FILE = DATASET_ROOT / "deprecated" / "hard_negative_mining_summary.json"


def select_device():
    if torch.cuda.is_available():
        return 0

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=285,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def read_pool():
    lines = [
        line.strip()
        for line in POOL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(lines) != len(set(lines)):
        raise ValueError("Hard-negative candidate list contains duplicate paths.")

    records = []

    for line in lines:
        image_path = (DATASET_ROOT / line).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        label_path = (
            DATASET_ROOT
            / "labels"
            / "train"
            / f"{image_path.stem}.txt"
        )

        if not label_path.is_file():
            raise FileNotFoundError(f"Label not found: {label_path}")

        if label_path.read_text(encoding="utf-8").strip():
            raise ValueError(
                f"Candidate negative image has a non-empty label: {image_path.name}"
            )

        records.append(
            {
                "line": line,
                "image_path": image_path,
            }
        )

    return records


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    args = parse_args()
    device = args.device or select_device()
    model_path = args.model.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not POOL_FILE.is_file():
        raise FileNotFoundError(f"Candidate list not found: {POOL_FILE}")

    records = read_pool()
    line_by_path = {
        str(record["image_path"]): record["line"]
        for record in records
    }

    model = YOLO(str(model_path))
    scored = []
    processed = 0

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Candidate negative images: {len(records)}")

    image_paths = [
        str(record["image_path"])
        for record in records
    ]

    for chunk in chunks(image_paths, args.chunk_size):
        results = model.predict(
            source=chunk,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            conf=0.001,
            iou=0.70,
            max_det=50,
            save=False,
            verbose=False,
            stream=True,
        )

        for source_path, result in zip(chunk, results):
            confidences = []

            if result.boxes is not None and len(result.boxes):
                confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .tolist()
                )

            source_path = str(Path(source_path).resolve())
            image_line = line_by_path[source_path]

            scored.append(
                {
                    "image": image_line,
                    "max_confidence": (
                        max(confidences) if confidences else 0.0
                    ),
                    "detections_ge_005": sum(
                        value >= 0.05 for value in confidences
                    ),
                    "detections_ge_010": sum(
                        value >= 0.10 for value in confidences
                    ),
                    "detections_ge_025": sum(
                        value >= 0.25 for value in confidences
                    ),
                    "detections_ge_050": sum(
                        value >= 0.50 for value in confidences
                    ),
                }
            )

            processed += 1

        print(
            f"\rProcessed: {processed}/{len(records)}",
            end="",
            flush=True,
        )

    print()

    if processed != len(records):
        raise RuntimeError(
            f"Unexpected result count: expected {len(records)}, got {processed}"
        )

    scored.sort(
        key=lambda row: (
            -row["max_confidence"],
            row["image"],
        )
    )

    selected_count = min(args.count, len(scored))
    selected = scored[:selected_count]

    with SCORES_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "max_confidence",
                "detections_ge_005",
                "detections_ge_010",
                "detections_ge_025",
                "detections_ge_050",
            ],
        )
        writer.writeheader()
        writer.writerows(scored)

    SELECTED_FILE.write_text(
        "\n".join(row["image"] for row in selected) + "\n",
        encoding="utf-8",
    )

    summary = {
        "model": str(model_path),
        "candidate_negative_images": len(scored),
        "selected_hard_negatives": selected_count,
        "imgsz": args.imgsz,
        "images_with_confidence_ge_005": sum(
            row["max_confidence"] >= 0.05 for row in scored
        ),
        "images_with_confidence_ge_010": sum(
            row["max_confidence"] >= 0.10 for row in scored
        ),
        "images_with_confidence_ge_025": sum(
            row["max_confidence"] >= 0.25 for row in scored
        ),
        "images_with_confidence_ge_050": sum(
            row["max_confidence"] >= 0.50 for row in scored
        ),
        "highest_false_positive_confidence": (
            scored[0]["max_confidence"] if scored else 0.0
        ),
        "lowest_selected_confidence": (
            selected[-1]["max_confidence"] if selected else 0.0
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"All scores: {SCORES_FILE}")
    print(f"Selected hard negatives: {SELECTED_FILE}")
    print("No images, labels, validation data, or test data were modified.")


if __name__ == "__main__":
    main()
