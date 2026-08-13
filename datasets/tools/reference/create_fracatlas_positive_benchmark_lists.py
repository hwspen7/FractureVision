import json
from pathlib import Path


DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "fracatlas_yolo"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

EXPECTED = {
    "train": {
        "images": 570,
        "objects": 757,
    },
    "val": {
        "images": 82,
        "objects": 91,
    },
    "test": {
        "images": 63,
        "objects": 69,
    },
}


def create_positive_list(split):
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}"
        )

    images_by_stem = {}

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.stem in images_by_stem:
            raise RuntimeError(
                f"{split} has duplicate image stem: "
                f"{image_path.stem}"
            )

        images_by_stem[image_path.stem] = image_path

    positive_lines = []
    object_count = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        rows = [
            row.strip()
            for row in label_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            if row.strip()
        ]

        if not rows:
            continue

        image_path = images_by_stem.get(label_path.stem)

        if image_path is None:
            raise FileNotFoundError(
                f"Positive label has no matching image: {label_path}"
            )

        for line_number, row in enumerate(rows, start=1):
            parts = row.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid label format: "
                    f"{label_path}:{line_number}"
                )

            class_id, x, y, width, height = map(
                float,
                parts,
            )

            if int(class_id) != 0:
                raise ValueError(
                    f"Nonzero class found: "
                    f"{label_path}:{line_number}"
                )

            if not all(
                0.0 <= value <= 1.0
                for value in (x, y, width, height)
            ):
                raise ValueError(
                    f"Coordinates out of range: "
                    f"{label_path}:{line_number}"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid box width or height: "
                    f"{label_path}:{line_number}"
                )

        positive_lines.append(
            f"./images/{split}/{image_path.name}"
        )
        object_count += len(rows)

    expected = EXPECTED[split]

    if len(positive_lines) != expected["images"]:
        raise RuntimeError(
            f"{split} should have {expected['images']} positive images, "
            f"got {len(positive_lines)}."
        )

    if object_count != expected["objects"]:
        raise RuntimeError(
            f"{split} should have {expected['objects']} boxes, "
            f"got {object_count}."
        )

    output_path = (
        DATASET_ROOT
        / f"benchmark_{split}_positive.txt"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Output list already exists: {output_path}"
        )

    output_path.write_text(
        "\n".join(positive_lines) + "\n",
        encoding="utf-8",
    )

    return {
        "split": split,
        "positive_images": len(positive_lines),
        "objects": object_count,
        "output": str(output_path),
        "stems": {
            Path(line).stem
            for line in positive_lines
        },
    }


def main():
    results = [
        create_positive_list(split)
        for split in ("train", "val", "test")
    ]

    split_stems = {
        result["split"]: result.pop("stems")
        for result in results
    }

    overlaps = {
        "train_val": len(
            split_stems["train"]
            & split_stems["val"]
        ),
        "train_test": len(
            split_stems["train"]
            & split_stems["test"]
        ),
        "val_test": len(
            split_stems["val"]
            & split_stems["test"]
        ),
    }

    if any(overlaps.values()):
        raise RuntimeError(
            f"Positive benchmark splits overlap: {overlaps}"
        )

    summary = {
        "splits": results,
        "overlaps": overlaps,
    }

    print(json.dumps(summary, indent=2))
    print(
        "Only positive benchmark lists were generated, "
        "images, labels, and dataset splits were not modified."
    )


if __name__ == "__main__":
    main()
