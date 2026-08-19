import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "datasets" / "detection"

MODEL_PATH = (
    ROOT
    / "experiments"
    / "detection"
    / "training"
    / "baseline"
    / "weights"
    / "best.pt"
)

SCORES_FILE = DATASET_ROOT / "deprecated" / "hard_negative_scores.csv"
OUTPUT_DIR = DATASET_ROOT / "deprecated" / "hard_negative_review"

COLS = 3
ROWS = 2
IMAGES_PER_PAGE = COLS * ROWS

TILE_WIDTH = 500
TILE_HEIGHT = 500
TITLE_HEIGHT = 36


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
        "--threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        default=None,
    )

    return parser.parse_args()


def load_candidates(threshold, max_images):
    with SCORES_FILE.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    candidates = [
        row for row in rows
        if float(row["max_confidence"]) >= threshold
    ]

    candidates.sort(
        key=lambda row: (
            -float(row["max_confidence"]),
            row["image"],
        )
    )

    return candidates[:max_images]


def fit_image(image):
    image = image.copy()
    image.thumbnail(
        (TILE_WIDTH, TILE_HEIGHT),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (TILE_WIDTH, TILE_HEIGHT),
        "black",
    )

    x = (TILE_WIDTH - image.width) // 2
    y = (TILE_HEIGHT - image.height) // 2
    canvas.paste(image, (x, y))

    return canvas


def save_pages(items):
    page_width = COLS * TILE_WIDTH
    page_height = ROWS * (TITLE_HEIGHT + TILE_HEIGHT)

    index_rows = []

    for page_start in range(0, len(items), IMAGES_PER_PAGE):
        page_items = items[
            page_start:page_start + IMAGES_PER_PAGE
        ]

        page_number = page_start // IMAGES_PER_PAGE + 1

        page = Image.new(
            "RGB",
            (page_width, page_height),
            "#202020",
        )
        draw = ImageDraw.Draw(page)

        for slot, item in enumerate(page_items):
            row = slot // COLS
            col = slot % COLS

            x = col * TILE_WIDTH
            y = row * (TITLE_HEIGHT + TILE_HEIGHT)

            confidence = float(item["max_confidence"])
            title = (
                f"{Path(item['image']).name} | "
                f"negative label | "
                f"model {confidence:.1%}"
            )

            draw.rectangle(
                (
                    x,
                    y,
                    x + TILE_WIDTH,
                    y + TITLE_HEIGHT,
                ),
                fill="#303030",
            )
            draw.text(
                (x + 8, y + 10),
                title,
                fill="#FFD166",
            )

            page.paste(
                item["annotated_image"],
                (x, y + TITLE_HEIGHT),
            )

            index_rows.append(
                {
                    "page": page_number,
                    "slot": slot + 1,
                    "image": item["image"],
                    "max_confidence": confidence,
                }
            )

        page_path = OUTPUT_DIR / f"page_{page_number:03d}.jpg"
        page.save(
            page_path,
            quality=95,
            subsampling=0,
        )

    with (OUTPUT_DIR / "review_index.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "page",
                "slot",
                "image",
                "max_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)


def main():
    args = parse_args()
    device = args.device or select_device()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not SCORES_FILE.is_file():
        raise FileNotFoundError(f"Score file not found: {SCORES_FILE}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists, check or rename it first: {OUTPUT_DIR}"
        )

    candidates = load_candidates(
        threshold=args.threshold,
        max_images=args.max_images,
    )

    if not candidates:
        raise RuntimeError("No hard-negative images meet the threshold.")

    OUTPUT_DIR.mkdir(parents=True)

    image_paths = [
        str((DATASET_ROOT / row["image"]).resolve())
        for row in candidates
    ]

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=image_paths,
        imgsz=768,
        batch=4,
        device=device,
        conf=0.05,
        iou=0.70,
        max_det=50,
        save=False,
        verbose=False,
        stream=True,
    )

    review_items = []

    for row, result in zip(candidates, results):
        plotted = result.plot(
            labels=True,
            conf=True,
            line_width=2,
        )

        rgb_image = Image.fromarray(
            plotted[:, :, ::-1]
        )

        review_items.append(
            {
                **row,
                "annotated_image": fit_image(rgb_image),
            }
        )

    if len(review_items) != len(candidates):
        raise RuntimeError(
            "Prediction count does not match candidate image count."
        )

    save_pages(review_items)

    print(f"Review image count: {len(review_items)}")
    print(f"Review page count: {len(list(OUTPUT_DIR.glob('page_*.jpg')))}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("No training images, labels, validation data, or test data were modified.")


if __name__ == "__main__":
    main()
