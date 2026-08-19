import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "segmentation"
IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "segmentation"
    / "training"
    / "positive_only"
    / "weights"
    / "best.pt"
)

SCORES_FILE = DATASET_ROOT / "hard_negative_scores.csv"
SELECTED_FILE = DATASET_ROOT / "mined_hard_negatives.txt"
SUMMARY_FILE = DATASET_ROOT / "hard_negative_mining_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def select_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine false-positive FracAtlas training negatives with the segmentation model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--count", type=int, default=570)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def read_negative_candidates():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Missing training image directory: {IMAGE_DIR}")
    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Missing training label directory: {LABEL_DIR}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    records = []
    positive_count = 0

    for image_path in images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path}")

        if label_path.read_text(encoding="utf-8", errors="ignore").strip():
            positive_count += 1
            continue

        records.append(
            {
                "line": image_path.relative_to(DATASET_ROOT).as_posix(),
                "image_path": image_path.resolve(),
            }
        )

    if positive_count != 570:
        raise RuntimeError(f"Expected 570 positive training images, found {positive_count}.")
    if len(records) != 2678:
        raise RuntimeError(f"Expected 2678 negative training images, found {len(records)}.")

    return records


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    args = parse_args()
    device = args.device or select_device()
    model_path = args.model.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if args.count <= 0:
        raise ValueError("--count must be positive.")

    records = read_negative_candidates()
    model = YOLO(str(model_path))
    if model.task != "segment":
        raise RuntimeError(f"Expected a segmentation model, found task={model.task!r}.")

    image_paths = [str(record["image_path"]) for record in records]
    line_by_path = {
        str(record["image_path"]): record["line"] for record in records
    }
    scored = []
    processed = 0

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Candidate negative images: {len(records)}")

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
                confidences = result.boxes.conf.detach().cpu().tolist()

            resolved_source = str(Path(source_path).resolve())
            scored.append(
                {
                    "image": line_by_path[resolved_source],
                    "max_confidence": max(confidences) if confidences else 0.0,
                    "detections_ge_005": sum(value >= 0.05 for value in confidences),
                    "detections_ge_010": sum(value >= 0.10 for value in confidences),
                    "detections_ge_025": sum(value >= 0.25 for value in confidences),
                    "detections_ge_050": sum(value >= 0.50 for value in confidences),
                }
            )
            processed += 1

        print(f"\rProcessed: {processed}/{len(records)}", end="", flush=True)

    print()
    if processed != len(records):
        raise RuntimeError(f"Expected {len(records)} results, found {processed}.")

    scored.sort(key=lambda row: (-row["max_confidence"], row["image"]))
    selected = scored[: min(args.count, len(scored))]

    with SCORES_FILE.open("w", encoding="utf-8", newline="") as file:
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
        "selected_hard_negatives": len(selected),
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
    print(f"\nAll scores: {SCORES_FILE}")
    print(f"Selected hard negatives: {SELECTED_FILE}")
    print("No images, labels, validation data, test data, or model weights were modified.")


if __name__ == "__main__":
    main()
