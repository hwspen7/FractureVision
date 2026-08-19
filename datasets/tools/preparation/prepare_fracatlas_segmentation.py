from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


DATASETS_DIR = Path(__file__).resolve().parents[2]
SOURCE_DETECTION = DATASETS_DIR / "detection"
SOURCE_RAW = DATASETS_DIR / "raw" / "FracAtlas"
DEFAULT_OUTPUT = DATASETS_DIR / "segmentation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the cleaned FracAtlas split to YOLO segmentation format."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Directory containing detection and raw datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to datasets/segmentation.",
    )
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing detection label: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def detection_box(line: str, path: Path) -> tuple[float, float, float, float]:
    parts = line.split()
    if len(parts) != 5 or parts[0] != "0":
        raise ValueError(f"Invalid detection label row: {path}: {line}")

    try:
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError as error:
        raise ValueError(f"Non-numeric detection label row: {path}: {line}") from error

    if not (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    ):
        raise ValueError(f"Out-of-range detection label row: {path}: {line}")

    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def coco_box(annotation: dict, image_info: dict) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, annotation["bbox"])
    image_width = float(image_info["width"])
    image_height = float(image_info["height"])
    return (
        x / image_width,
        y / image_height,
        (x + width) / image_width,
        (y + height) / image_height,
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def verify_boxes(
    label_lines: list[str],
    annotations: list[dict],
    image_info: dict,
    label_path: Path,
) -> list[float]:
    if len(label_lines) != len(annotations):
        raise ValueError(
            f"Object-count mismatch for {image_info['file_name']}: "
            f"detection={len(label_lines)}, segmentation={len(annotations)}"
        )

    detection_boxes = [detection_box(line, label_path) for line in label_lines]
    segmentation_boxes = [coco_box(annotation, image_info) for annotation in annotations]
    remaining = set(range(len(segmentation_boxes)))
    matched_ious: list[float] = []

    for current_box in detection_boxes:
        best_index = max(
            remaining,
            key=lambda index: box_iou(current_box, segmentation_boxes[index]),
        )
        current_iou = box_iou(current_box, segmentation_boxes[best_index])
        remaining.remove(best_index)

        if current_iou < 0.99:
            raise ValueError(
                f"Detection/segmentation box mismatch for {image_info['file_name']}: "
                f"IoU={current_iou:.6f}"
            )

        matched_ious.append(current_iou)

    return matched_ious


def polygon_to_yolo(annotation: dict, image_info: dict) -> tuple[str, int]:
    segmentation = annotation.get("segmentation")

    if not isinstance(segmentation, list) or len(segmentation) != 1:
        raise ValueError(
            f"Expected one polygon for annotation {annotation.get('id')}, "
            f"found {0 if not isinstance(segmentation, list) else len(segmentation)}"
        )

    polygon = segmentation[0]
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        raise ValueError(f"Invalid polygon for annotation {annotation.get('id')}")

    image_width = float(image_info["width"])
    image_height = float(image_info["height"])
    normalized: list[float] = []

    for index in range(0, len(polygon), 2):
        x = float(polygon[index]) / image_width
        y = float(polygon[index + 1]) / image_height

        if not (-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6):
            raise ValueError(
                f"Polygon point outside image for annotation {annotation.get('id')}: "
                f"x={x}, y={y}"
            )

        normalized.extend((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))

    unique_points = {
        (round(normalized[index], 8), round(normalized[index + 1], 8))
        for index in range(0, len(normalized), 2)
    }
    if len(unique_points) < 3:
        raise ValueError(f"Polygon has fewer than three unique points: {annotation.get('id')}")

    values = " ".join(f"{value:.8f}" for value in normalized)
    return f"0 {values}", len(normalized) // 2


def verify_display_aspect_ratio(image_path: Path, image_info: dict) -> None:
    with Image.open(image_path) as image:
        displayed = ImageOps.exif_transpose(image)
        actual_width, actual_height = displayed.size

    actual_ratio = actual_width / actual_height
    annotation_ratio = float(image_info["width"]) / float(image_info["height"])
    relative_error = abs(actual_ratio - annotation_ratio) / annotation_ratio

    if relative_error > 0.01:
        raise ValueError(
            f"Image/annotation aspect-ratio mismatch for {image_path.name}: "
            f"displayed={actual_width}x{actual_height}, "
            f"COCO={image_info['width']}x{image_info['height']}"
        )


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    args = parse_args()
    datasets_dir = args.datasets_dir.resolve()
    source_detection = datasets_dir / "detection"
    source_raw = datasets_dir / "raw" / "FracAtlas"
    output = (args.output or datasets_dir / "segmentation").resolve()
    building = output.with_name(f"{output.name}_building")

    if output.exists():
        raise FileExistsError(f"Output already exists and was not overwritten: {output}")
    if building.exists():
        raise FileExistsError(
            f"Incomplete build directory exists: {building}\n"
            "Inspect or rename it before trying again."
        )

    coco_path = source_raw / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO segmentation file: {coco_path}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id = {image["id"]: image for image in coco["images"]}
    images_by_name = {image["file_name"]: image for image in coco["images"]}
    annotations_by_name: dict[str, list[dict]] = defaultdict(list)

    for annotation in coco["annotations"]:
        image_info = images_by_id.get(annotation["image_id"])
        if image_info is None:
            raise ValueError(f"Annotation references missing image: {annotation.get('id')}")
        if annotation.get("category_id") != 1:
            raise ValueError(f"Unexpected category ID: {annotation.get('category_id')}")
        if annotation.get("iscrowd", 0) != 0:
            raise ValueError(f"Crowd/RLE annotation is not supported: {annotation.get('id')}")
        annotations_by_name[image_info["file_name"]].append(annotation)

    for annotations in annotations_by_name.values():
        annotations.sort(key=lambda item: item["id"])

    split_summary: dict[str, dict] = {}
    all_ious: list[float] = []
    link_modes: Counter[str] = Counter()
    included_positive_names: set[str] = set()

    for split in SPLITS:
        source_image_dir = source_detection / "images" / split
        source_label_dir = source_detection / "labels" / split
        target_image_dir = building / "images" / split
        target_label_dir = building / "labels" / split
        target_image_dir.mkdir(parents=True, exist_ok=False)
        target_label_dir.mkdir(parents=True, exist_ok=False)

        image_paths = sorted(
            path
            for path in source_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_stems = {path.stem for path in image_paths}
        orphan_labels = sorted(
            path.name
            for path in source_label_dir.glob("*.txt")
            if path.stem not in image_stems
        )
        if orphan_labels:
            raise ValueError(f"Orphan labels in {split}: {orphan_labels[:10]}")

        positive_images = 0
        negative_images = 0
        objects = 0
        polygon_points = 0
        positive_output_paths: list[str] = []

        for image_path in image_paths:
            source_label = source_label_dir / f"{image_path.stem}.txt"
            label_lines = read_nonempty_lines(source_label)
            image_info = images_by_name.get(image_path.name)
            annotations = annotations_by_name.get(image_path.name, [])

            if bool(label_lines) != bool(annotations):
                raise ValueError(
                    f"Positive/negative mismatch for {image_path.name}: "
                    f"detection_objects={len(label_lines)}, segmentation_objects={len(annotations)}"
                )

            target_image = target_image_dir / image_path.name
            target_label = target_label_dir / f"{image_path.stem}.txt"
            link_modes[link_or_copy(image_path, target_image)] += 1

            segmentation_lines: list[str] = []
            if annotations:
                if image_info is None:
                    raise ValueError(f"Missing COCO image metadata: {image_path.name}")

                verify_display_aspect_ratio(image_path, image_info)
                all_ious.extend(verify_boxes(label_lines, annotations, image_info, source_label))

                for annotation in annotations:
                    segmentation_line, point_count = polygon_to_yolo(annotation, image_info)
                    segmentation_lines.append(segmentation_line)
                    polygon_points += point_count

                positive_images += 1
                objects += len(segmentation_lines)
                included_positive_names.add(image_path.name)
                positive_output_paths.append(
                    str((output / "images" / split / image_path.name).resolve())
                )
            else:
                negative_images += 1

            target_label.write_text(
                "\n".join(segmentation_lines) + ("\n" if segmentation_lines else ""),
                encoding="utf-8",
            )

        (building / f"{split}_positive.txt").write_text(
            "\n".join(positive_output_paths) + "\n",
            encoding="utf-8",
        )

        split_summary[split] = {
            "images": len(image_paths),
            "positive": positive_images,
            "negative": negative_images,
            "objects": objects,
            "polygon_points": polygon_points,
        }

    expected_positive_names = set(annotations_by_name)
    excluded_coco_positives = sorted(expected_positive_names - included_positive_names)
    unexpected_prepared_positives = sorted(included_positive_names - expected_positive_names)
    if unexpected_prepared_positives:
        raise ValueError(
            f"Prepared positives absent from COCO annotations: {unexpected_prepared_positives}"
        )

    if (source_detection / "split_manifest.csv").exists():
        shutil.copy2(source_detection / "split_manifest.csv", building / "split_manifest.csv")
    if (source_detection / "removed_duplicates.csv").exists():
        shutil.copy2(
            source_detection / "removed_duplicates.csv",
            building / "removed_duplicates.csv",
        )

    sorted_ious = sorted(all_ious)
    summary = {
        "source_detection_dataset": str(source_detection),
        "source_coco_segmentation": str(coco_path),
        "output_dataset": str(output),
        "class_names": {"0": "fracture"},
        "coco_positive_images": len(expected_positive_names),
        "coco_annotations": len(coco["annotations"]),
        "included_positive_images": len(included_positive_names),
        "included_segmentation_objects": len(all_ious),
        "excluded_duplicate_positive_ids": excluded_coco_positives,
        "box_alignment_iou": {
            "minimum": min(sorted_ious),
            "median": sorted_ious[len(sorted_ious) // 2],
            "matches_at_least_0_99": sum(value >= 0.99 for value in sorted_ious),
        },
        "image_storage": dict(sorted(link_modes.items())),
        "splits": split_summary,
    }
    (building / "segmentation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    building.rename(output)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSegmentation dataset generated: {output}")
    print("Existing Detect data, raw images, labels, validation data, and test data were not modified.")


if __name__ == "__main__":
    main()
