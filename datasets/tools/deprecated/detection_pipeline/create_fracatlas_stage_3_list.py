import hashlib
import json
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = DATASETS_DIR / "fracatlas_yolo"

STAGE_2_LIST = DATASET_ROOT / "train_stage2.txt"
STAGE_3_LIST = DATASET_ROOT / "train_stage3.txt"

SEED = 44


def deterministic_key(image_line, copy_index):
    """
    Produces a deterministic shuffle key.

    copy_index distinguishes repeated positive-image entries.
    """
    value = f"{SEED}|{copy_index}|{image_line}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_label_path(image_path):
    """
    Converts an image path under images/train into its YOLO label path.
    """
    return (
        DATASET_ROOT
        / "labels"
        / "train"
        / f"{image_path.stem}.txt"
    )


def main():
    if not STAGE_2_LIST.exists():
        raise FileNotFoundError(
            f"Stage-2 training list not found: {STAGE_2_LIST}"
        )

    image_lines = [
        line.strip()
        for line in STAGE_2_LIST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(image_lines) != len(set(image_lines)):
        raise RuntimeError(
            "Stage-2 list already contains duplicate image paths."
        )

    positive_lines = []
    negative_lines = []

    for image_line in image_lines:
        image_path = Path(image_line)

        if not image_path.is_absolute():
            image_path = DATASET_ROOT / image_path

        image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        label_path = get_label_path(image_path)

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found: {label_path}"
            )

        label_content = label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if label_content:
            positive_lines.append(image_line)
        else:
            negative_lines.append(image_line)

    if len(positive_lines) != 570:
        raise RuntimeError(
            f"Expected 570 positive images, got "
            f"{len(positive_lines)}."
        )

    if len(negative_lines) != 570:
        raise RuntimeError(
            f"Expected 570 negative images, got "
            f"{len(negative_lines)}."
        )

    # Every positive image is exposed twice.
    # Every negative image is exposed once.
    training_entries = []

    for image_line in positive_lines:
        training_entries.append((image_line, 0))
        training_entries.append((image_line, 1))

    for image_line in negative_lines:
        training_entries.append((image_line, 0))

    training_entries.sort(
        key=lambda item: deterministic_key(
            item[0],
            item[1],
        )
    )

    STAGE_3_LIST.write_text(
        "\n".join(
            image_line
            for image_line, _ in training_entries
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "unique_positive_images": len(positive_lines),
        "unique_negative_images": len(negative_lines),
        "positive_exposures": len(positive_lines) * 2,
        "negative_exposures": len(negative_lines),
        "total_training_exposures": len(training_entries),
        "positive_to_negative_exposure_ratio": "2:1",
        "seed": SEED,
    }

    print(json.dumps(summary, indent=2))
    print(f"\nStage-3 training list: {STAGE_3_LIST}")
    print("No images, labels, validation data, or test data were modified.")


if __name__ == "__main__":
    main()