import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = ROOT / "datasets" / "detection"

STAGE_2_LIST = DATASET_ROOT / "deprecated" / "train_stage2.txt"
HARD_NEGATIVE_LIST = DATASET_ROOT / "deprecated" / "mined_hard_negatives.txt"
RANDOM_NEGATIVE_LIST = DATASET_ROOT / "deprecated" / "stage2_random_negatives.txt"

BUILD_ROOT = DATASET_ROOT / "deprecated" / "stage3_crops_building"
OUTPUT_ROOT = DATASET_ROOT / "deprecated" / "stage3_crops"

OUTPUT_IMAGES = BUILD_ROOT / "images" / "train"
OUTPUT_LABELS = BUILD_ROOT / "labels" / "train"

TRAIN_LIST = DATASET_ROOT / "deprecated" / "train_stage3_crops.txt"
SUMMARY_FILE = BUILD_ROOT / "summary.json"

DEFAULT_MODEL = (
    ROOT
    / "experiments"
    / "detection"
    / "training"
    / "balanced_hard_negative"
    / "weights"
    / "best.pt"
)

SEED = 45
IMAGE_QUALITY = 95


def select_device(requested):
    if requested:
        return requested

    if torch.cuda.is_available():
        return 0

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def stable_seed(text):
    digest = hashlib.sha256(
        f"{SEED}|{text}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def read_lines(path):
    if not path.is_file():
        raise FileNotFoundError(f"List file not found: {path}")

    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(lines) != len(set(lines)):
        raise ValueError(f"List contains duplicate paths: {path}")

    return lines


def resolve_image(image_line):
    image_path = Path(image_line)

    if not image_path.is_absolute():
        image_path = DATASET_ROOT / image_path

    image_path = image_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def source_label_path(image_path):
    return (
        DATASET_ROOT
        / "labels"
        / "train"
        / f"{image_path.stem}.txt"
    )


def read_yolo_labels(image_path):
    label_path = source_label_path(image_path)

    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

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

        labels.append(
            {
                "class_id": 0,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    return labels


def load_image(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def label_to_xyxy(label, image_width, image_height):
    center_x = label["x"] * image_width
    center_y = label["y"] * image_height
    box_width = label["width"] * image_width
    box_height = label["height"] * image_height

    return (
        center_x - box_width / 2,
        center_y - box_height / 2,
        center_x + box_width / 2,
        center_y + box_height / 2,
    )


def make_context_crop(
    center_x,
    center_y,
    target_width,
    target_height,
    image_width,
    image_height,
    rng,
    jitter=True,
):
    """
    Generates a square contextual crop around a target region.

    The crop is large enough to retain anatomical context while making
    small fracture regions occupy more feature-map cells.
    """
    shortest_side = min(image_width, image_height)

    minimum_side = shortest_side * 0.35
    preferred_side = max(target_width, target_height) * 6.0
    maximum_side = shortest_side * 0.80

    crop_side = max(minimum_side, preferred_side)
    crop_side = min(crop_side, maximum_side)

    # Always make sure the target itself fits inside the crop.
    crop_side = max(
        crop_side,
        target_width * 1.30,
        target_height * 1.30,
    )
    crop_side = min(crop_side, shortest_side)
    crop_side = max(32, int(round(crop_side)))

    if jitter:
        center_x += rng.uniform(-0.05, 0.05) * crop_side
        center_y += rng.uniform(-0.05, 0.05) * crop_side

    left = int(round(center_x - crop_side / 2))
    top = int(round(center_y - crop_side / 2))

    left = max(0, min(left, image_width - crop_side))
    top = max(0, min(top, image_height - crop_side))

    return (
        left,
        top,
        left + crop_side,
        top + crop_side,
    )


def make_random_crop(image_width, image_height, rng):
    shortest_side = min(image_width, image_height)

    crop_side = int(
        round(
            shortest_side
            * rng.uniform(0.45, 0.75)
        )
    )
    crop_side = max(32, min(crop_side, shortest_side))

    left = rng.randint(0, max(0, image_width - crop_side))
    top = rng.randint(0, max(0, image_height - crop_side))

    return (
        left,
        top,
        left + crop_side,
        top + crop_side,
    )


def transform_labels(
    labels,
    crop_box,
    image_width,
    image_height,
):
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    transformed = []

    for label in labels:
        x1, y1, x2, y2 = label_to_xyxy(
            label,
            image_width,
            image_height,
        )

        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        clipped_x1 = max(x1, crop_left)
        clipped_y1 = max(y1, crop_top)
        clipped_x2 = min(x2, crop_right)
        clipped_y2 = min(y2, crop_bottom)

        visible_width = max(0.0, clipped_x2 - clipped_x1)
        visible_height = max(0.0, clipped_y2 - clipped_y1)
        visible_area = visible_width * visible_height

        if original_area <= 0:
            continue

        # Discard heavily truncated boxes.
        if visible_area / original_area < 0.70:
            continue

        local_x1 = clipped_x1 - crop_left
        local_y1 = clipped_y1 - crop_top
        local_x2 = clipped_x2 - crop_left
        local_y2 = clipped_y2 - crop_top

        center_x = (local_x1 + local_x2) / 2 / crop_width
        center_y = (local_y1 + local_y2) / 2 / crop_height
        width = (local_x2 - local_x1) / crop_width
        height = (local_y2 - local_y1) / crop_height

        if width <= 0 or height <= 0:
            continue

        transformed.append(
            {
                "class_id": 0,
                "x": min(max(center_x, 0.0), 1.0),
                "y": min(max(center_y, 0.0), 1.0),
                "width": min(max(width, 0.0), 1.0),
                "height": min(max(height, 0.0), 1.0),
            }
        )

    return transformed


def save_label(path, labels):
    lines = [
        (
            f"{label['class_id']} "
            f"{label['x']:.6f} "
            f"{label['y']:.6f} "
            f"{label['width']:.6f} "
            f"{label['height']:.6f}"
        )
        for label in labels
    ]

    content = "\n".join(lines)

    if content:
        content += "\n"

    path.write_text(content, encoding="utf-8")


def save_crop(image, crop_box, filename, labels):
    cropped_image = image.crop(crop_box)

    image_path = OUTPUT_IMAGES / filename
    label_path = OUTPUT_LABELS / f"{Path(filename).stem}.txt"

    cropped_image.save(
        image_path,
        format="JPEG",
        quality=IMAGE_QUALITY,
        subsampling=0,
    )
    save_label(label_path, labels)

    return (
        f"./deprecated/stage3_crops/images/train/{filename}"
    )


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


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
        "--batch",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    device = select_device(args.device)

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_ROOT}"
        )

    if BUILD_ROOT.exists():
        raise FileExistsError(
            f"Unfinished build directory found: {BUILD_ROOT}"
        )

    if TRAIN_LIST.exists():
        raise FileExistsError(
            f"Training list already exists: {TRAIN_LIST}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Stage 2 model not found: {model_path}"
        )

    stage_2_lines = read_lines(STAGE_2_LIST)
    hard_negative_lines = read_lines(HARD_NEGATIVE_LIST)
    random_negative_lines = read_lines(RANDOM_NEGATIVE_LIST)

    positive_records = []
    negative_records = []

    for image_line in stage_2_lines:
        image_path = resolve_image(image_line)
        labels = read_yolo_labels(image_path)

        record = {
            "line": image_line,
            "path": image_path,
            "labels": labels,
        }

        if labels:
            positive_records.append(record)
        else:
            negative_records.append(record)

    if len(positive_records) != 570:
        raise RuntimeError(
            f"Expected 570 positive source images, got "
            f"{len(positive_records)}."
        )

    if len(negative_records) != 570:
        raise RuntimeError(
            f"Expected 570 negative source images, got "
            f"{len(negative_records)}."
        )

    stage_2_negative_paths = {
        str(record["path"])
        for record in negative_records
    }
    hard_negative_paths = {
        str(resolve_image(line))
        for line in hard_negative_lines
    }
    random_negative_paths = {
        str(resolve_image(line))
        for line in random_negative_lines
    }

    if hard_negative_paths & random_negative_paths:
        raise RuntimeError(
            "Hard negatives and random negatives overlap."
        )

    if (
        hard_negative_paths | random_negative_paths
        != stage_2_negative_paths
    ):
        raise RuntimeError(
            "Stage 2 negatives do not match the hard and random negative sets."
        )

    if len(hard_negative_lines) != 285:
        raise RuntimeError("Hard-negative count is not 285.")

    if len(random_negative_lines) != 285:
        raise RuntimeError("Random-negative count is not 285.")

    OUTPUT_IMAGES.mkdir(parents=True)
    OUTPUT_LABELS.mkdir(parents=True)

    crop_lines = []
    positive_crop_count = 0
    positive_crop_objects = 0

    print("Generating positive fracture context crops.")

    for record_index, record in enumerate(
        positive_records,
        start=1,
    ):
        image = load_image(record["path"])
        image_width, image_height = image.size

        for box_index, target_label in enumerate(
            record["labels"]
        ):
            rng = random.Random(
                stable_seed(
                    f"positive|{record['path']}|{box_index}"
                )
            )

            x1, y1, x2, y2 = label_to_xyxy(
                target_label,
                image_width,
                image_height,
            )

            crop_box = make_context_crop(
                center_x=(x1 + x2) / 2,
                center_y=(y1 + y2) / 2,
                target_width=x2 - x1,
                target_height=y2 - y1,
                image_width=image_width,
                image_height=image_height,
                rng=rng,
                jitter=True,
            )

            transformed_labels = transform_labels(
                record["labels"],
                crop_box,
                image_width,
                image_height,
            )

            if not transformed_labels:
                raise RuntimeError(
                    f"Positive crop retained no boxes: "
                    f"{record['path'].name} #{box_index}"
                )

            filename = (
                f"pos_{record['path'].stem}"
                f"_box_{box_index:02d}.jpg"
            )

            crop_lines.append(
                save_crop(
                    image,
                    crop_box,
                    filename,
                    transformed_labels,
                )
            )

            positive_crop_count += 1
            positive_crop_objects += len(
                transformed_labels
            )

        if record_index % 50 == 0:
            print(
                f"Positive source progress: "
                f"{record_index}/{len(positive_records)}"
            )

    print("Locating and cropping hard-negative false-positive regions.")

    model = YOLO(str(model_path))
    hard_negative_crop_count = 0

    hard_records = [
        {
            "line": line,
            "path": resolve_image(line),
        }
        for line in hard_negative_lines
    ]

    chunk_size = max(1, args.batch * 4)

    for batch_index, batch_records in enumerate(
        chunks(hard_records, chunk_size),
        start=1,
    ):
        source_paths = [
            str(record["path"])
            for record in batch_records
        ]

        results = model.predict(
            source=source_paths,
            imgsz=768,
            batch=args.batch,
            device=device,
            conf=0.001,
            iou=0.70,
            max_det=50,
            save=False,
            verbose=False,
            stream=False,
        )

        if len(results) != len(batch_records):
            raise RuntimeError(
                "Hard-negative inference result count does not match."
            )

        for record, result in zip(batch_records, results):
            image = load_image(record["path"])
            image_width, image_height = image.size

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):
                best_index = int(
                    result.boxes.conf.argmax().item()
                )
                x1, y1, x2, y2 = (
                    result.boxes.xyxy[best_index]
                    .detach()
                    .cpu()
                    .tolist()
                )

                rng = random.Random(
                    stable_seed(
                        f"hard|{record['path']}"
                    )
                )

                crop_box = make_context_crop(
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    target_width=max(1.0, x2 - x1),
                    target_height=max(1.0, y2 - y1),
                    image_width=image_width,
                    image_height=image_height,
                    rng=rng,
                    jitter=False,
                )
            else:
                crop_side = int(
                    round(
                        min(image_width, image_height)
                        * 0.60
                    )
                )

                crop_box = (
                    (image_width - crop_side) // 2,
                    (image_height - crop_side) // 2,
                    (image_width - crop_side) // 2
                    + crop_side,
                    (image_height - crop_side) // 2
                    + crop_side,
                )

            filename = (
                f"hard_{record['path'].stem}.jpg"
            )

            crop_lines.append(
                save_crop(
                    image,
                    crop_box,
                    filename,
                    [],
                )
            )
            hard_negative_crop_count += 1

        processed = min(
            batch_index * chunk_size,
            len(hard_records),
        )
        print(
            f"\rHard-negative progress: "
            f"{processed}/{len(hard_records)}",
            end="",
            flush=True,
        )

    print()
    print("Generating random negative crops.")

    random_negative_crop_count = 0

    for index, image_line in enumerate(
        random_negative_lines,
        start=1,
    ):
        image_path = resolve_image(image_line)
        image = load_image(image_path)
        image_width, image_height = image.size

        rng = random.Random(
            stable_seed(f"random|{image_path}")
        )

        crop_box = make_random_crop(
            image_width,
            image_height,
            rng,
        )

        filename = f"random_{image_path.stem}.jpg"

        crop_lines.append(
            save_crop(
                image,
                crop_box,
                filename,
                [],
            )
        )
        random_negative_crop_count += 1

        if index % 50 == 0:
            print(
                f"Random-negative progress: "
                f"{index}/{len(random_negative_lines)}"
            )

    expected_positive_crops = sum(
        len(record["labels"])
        for record in positive_records
    )

    if positive_crop_count != expected_positive_crops:
        raise RuntimeError(
            "Positive crop count does not match training box count."
        )

    if hard_negative_crop_count != 285:
        raise RuntimeError(
            "Unexpected hard-negative crop count."
        )

    if random_negative_crop_count != 285:
        raise RuntimeError(
            "Unexpected random-negative crop count."
        )

    training_lines = stage_2_lines + crop_lines
    random.Random(SEED).shuffle(training_lines)

    if len(training_lines) != len(set(training_lines)):
        raise RuntimeError(
            "Stage 3 training list contains duplicate paths."
        )

    summary = {
        "source_full_images": len(stage_2_lines),
        "source_positive_images": len(positive_records),
        "source_negative_images": len(negative_records),
        "positive_context_crops": positive_crop_count,
        "objects_in_positive_crops": positive_crop_objects,
        "hard_negative_crops": hard_negative_crop_count,
        "random_negative_crops": random_negative_crop_count,
        "total_crop_images": len(crop_lines),
        "total_training_images": len(training_lines),
        "positive_training_images": (
            len(positive_records)
            + positive_crop_count
        ),
        "negative_training_images": (
            len(negative_records)
            + hard_negative_crop_count
            + random_negative_crop_count
        ),
        "seed": SEED,
        "model_used_for_hard_negative_crops": str(
            model_path
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    BUILD_ROOT.rename(OUTPUT_ROOT)

    TRAIN_LIST.write_text(
        "\n".join(training_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Crop data: {OUTPUT_ROOT}")
    print(f"Stage 3 training list: {TRAIN_LIST}")
    print(
        "Source images, source labels, validation data, and test data were not modified."
    )


if __name__ == "__main__":
    main()
