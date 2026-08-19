import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "datasets" / "detection"

CROP_ROOT = DATASET_ROOT / "deprecated" / "stage3_crops"
IMAGE_DIR = CROP_ROOT / "images" / "train"
LABEL_DIR = CROP_ROOT / "labels" / "train"

HARD_SCORES = DATASET_ROOT / "deprecated" / "hard_negative_scores.csv"
REVIEW_ROOT = DATASET_ROOT / "deprecated" / "stage3_crop_review"

SEED = 46
PAGE_COLUMNS = 4
PAGE_ROWS = 3
PAGE_CAPACITY = PAGE_COLUMNS * PAGE_ROWS

CELL_WIDTH = 450
CELL_HEIGHT = 390
HEADER_HEIGHT = 52


def load_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for candidate in candidates:
        path = Path(candidate)

        if path.is_file():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


FONT = load_font(17)
SMALL_FONT = load_font(14)


def read_labels(label_path):
    labels = []

    for line_number, line in enumerate(
        label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid label format: {label_path}:{line_number}"
            )

        class_id, x, y, width, height = map(float, parts)

        if int(class_id) != 0:
            raise ValueError(
                f"Nonzero class found: {label_path}:{line_number}"
            )

        if not all(
            0.0 <= value <= 1.0
            for value in (x, y, width, height)
        ):
            raise ValueError(
                f"Label coordinates out of range: {label_path}:{line_number}"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid label width or height: {label_path}:{line_number}"
            )

        labels.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )

    return labels


def classify_image(image_path):
    name = image_path.name

    if name.startswith("pos_"):
        return "positive"

    if name.startswith("hard_"):
        return "hard_negative"

    if name.startswith("random_"):
        return "random_negative"

    raise ValueError(f"Unknown crop type: {name}")


def load_hard_scores():
    scores = {}

    if not HARD_SCORES.is_file():
        return scores

    with HARD_SCORES.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            stem = Path(row["image"]).stem
            scores[stem] = float(row["max_confidence"])

    return scores


def build_records():
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    label_paths = sorted(LABEL_DIR.glob("*.txt"))

    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    if missing_labels:
        raise RuntimeError(
            f"Missing labels found: {missing_labels[:5]}"
        )

    if orphan_labels:
        raise RuntimeError(
            f"Orphan labels found: {orphan_labels[:5]}"
        )

    hard_scores = load_hard_scores()
    records = []

    for image_path in image_paths:
        category = classify_image(image_path)
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        labels = read_labels(label_path)

        if category == "positive" and not labels:
            raise RuntimeError(
                f"Positive crop has an empty label: {image_path.name}"
            )

        if category != "positive" and labels:
            raise RuntimeError(
                f"Negative crop contains target boxes: {image_path.name}"
            )

        with Image.open(image_path) as image:
            image.verify()

        source_stem = image_path.stem

        if category == "hard_negative":
            source_stem = source_stem.removeprefix("hard_")

        records.append(
            {
                "image_path": image_path,
                "category": category,
                "labels": labels,
                "min_area": min(
                    (label["area"] for label in labels),
                    default=0.0,
                ),
                "max_area": max(
                    (label["area"] for label in labels),
                    default=0.0,
                ),
                "hard_score": hard_scores.get(source_stem),
            }
        )

    return records


def choose_review_samples(records):
    positive = [
        record
        for record in records
        if record["category"] == "positive"
    ]
    hard_negative = [
        record
        for record in records
        if record["category"] == "hard_negative"
    ]
    random_negative = [
        record
        for record in records
        if record["category"] == "random_negative"
    ]

    rng = random.Random(SEED)

    # Positive review includes the smallest, largest,
    # and a deterministic random sample.
    sorted_positive = sorted(
        positive,
        key=lambda record: (
            record["min_area"],
            record["image_path"].name,
        ),
    )

    smallest = sorted_positive[:20]
    largest = sorted_positive[-20:]

    already_selected = {
        record["image_path"]
        for record in smallest + largest
    }

    remaining = [
        record
        for record in positive
        if record["image_path"] not in already_selected
    ]

    random_positive = rng.sample(
        remaining,
        min(20, len(remaining)),
    )

    selected_positive = smallest + largest + random_positive

    # Prioritize the strongest original false positives.
    selected_hard = sorted(
        hard_negative,
        key=lambda record: (
            -(
                record["hard_score"]
                if record["hard_score"] is not None
                else -1.0
            ),
            record["image_path"].name,
        ),
    )[:60]

    selected_random = rng.sample(
        random_negative,
        min(60, len(random_negative)),
    )

    return {
        "positive": selected_positive,
        "hard_negative": selected_hard,
        "random_negative": selected_random,
    }


def draw_record(record):
    with Image.open(record["image_path"]) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        original_width, original_height = source.size

        available_size = (
            CELL_WIDTH - 20,
            CELL_HEIGHT - HEADER_HEIGHT - 20,
        )

        thumbnail = ImageOps.contain(
            source,
            available_size,
            method=Image.Resampling.LANCZOS,
        )

    card = Image.new(
        "RGB",
        (CELL_WIDTH, CELL_HEIGHT),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(card)

    category = record["category"]

    if category == "positive":
        color = (70, 220, 110)
        detail = (
            f"objects={len(record['labels'])} | "
            f"min_area={record['min_area'] * 100:.2f}%"
        )
    elif category == "hard_negative":
        color = (255, 170, 40)
        score = record["hard_score"]

        detail = (
            "EMPTY LABEL | source FP="
            f"{score * 100:.1f}%"
            if score is not None
            else "EMPTY LABEL"
        )
    else:
        color = (170, 190, 210)
        detail = "EMPTY LABEL | RANDOM NEGATIVE"

    draw.text(
        (8, 6),
        record["image_path"].name,
        font=FONT,
        fill=color,
    )
    draw.text(
        (8, 29),
        detail,
        font=SMALL_FONT,
        fill=(220, 220, 220),
    )

    paste_x = (CELL_WIDTH - thumbnail.width) // 2
    paste_y = (
        HEADER_HEIGHT
        + (CELL_HEIGHT - HEADER_HEIGHT - thumbnail.height) // 2
    )

    card.paste(thumbnail, (paste_x, paste_y))
    draw = ImageDraw.Draw(card)

    for index, label in enumerate(record["labels"], start=1):
        center_x = label["x"] * thumbnail.width
        center_y = label["y"] * thumbnail.height
        box_width = label["width"] * thumbnail.width
        box_height = label["height"] * thumbnail.height

        x1 = paste_x + center_x - box_width / 2
        y1 = paste_y + center_y - box_height / 2
        x2 = paste_x + center_x + box_width / 2
        y2 = paste_y + center_y + box_height / 2

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3,
        )
        draw.text(
            (x1 + 3, max(paste_y, y1 - 20)),
            f"#{index} fracture",
            font=SMALL_FONT,
            fill=color,
        )

    return card


def render_pages(category, records):
    output_dir = REVIEW_ROOT / category
    output_dir.mkdir(parents=True)

    page_count = (
        len(records) + PAGE_CAPACITY - 1
    ) // PAGE_CAPACITY

    for page_index in range(page_count):
        page = Image.new(
            "RGB",
            (
                PAGE_COLUMNS * CELL_WIDTH,
                PAGE_ROWS * CELL_HEIGHT,
            ),
            (8, 8, 8),
        )

        page_records = records[
            page_index * PAGE_CAPACITY:
            (page_index + 1) * PAGE_CAPACITY
        ]

        for index, record in enumerate(page_records):
            row = index // PAGE_COLUMNS
            column = index % PAGE_COLUMNS

            card = draw_record(record)

            page.paste(
                card,
                (
                    column * CELL_WIDTH,
                    row * CELL_HEIGHT,
                ),
            )

        output_path = (
            output_dir / f"page_{page_index + 1:03d}.jpg"
        )

        page.save(
            output_path,
            quality=94,
            subsampling=0,
        )


def main():
    if not CROP_ROOT.is_dir():
        raise FileNotFoundError(
            f"Stage 3 crop data not found: {CROP_ROOT}"
        )

    if REVIEW_ROOT.exists():
        raise FileExistsError(
            f"Review directory already exists: {REVIEW_ROOT}"
        )

    records = build_records()
    samples = choose_review_samples(records)

    REVIEW_ROOT.mkdir(parents=True)

    for category, category_records in samples.items():
        render_pages(category, category_records)

    category_counts = {
        category: sum(
            record["category"] == category
            for record in records
        )
        for category in (
            "positive",
            "hard_negative",
            "random_negative",
        )
    }

    summary = {
        "checked_crop_images": len(records),
        "category_counts": category_counts,
        "positive_label_objects": sum(
            len(record["labels"])
            for record in records
            if record["category"] == "positive"
        ),
        "review_sample_counts": {
            category: len(category_records)
            for category, category_records in samples.items()
        },
        "page_counts": {
            category: (
                len(category_records)
                + PAGE_CAPACITY - 1
            ) // PAGE_CAPACITY
            for category, category_records in samples.items()
        },
        "seed": SEED,
    }

    (REVIEW_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Review pages: {REVIEW_ROOT}")
    print("No training images or labels were modified.")


if __name__ == "__main__":
    main()
