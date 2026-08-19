import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "datasets" / "configs" / "current" / "fracatlas_seg_recall_optimized.yaml"
DATASET_ROOT = ROOT / "datasets" / "segmentation"
TRAIN_LIST = DATASET_ROOT / "train_recall_optimized.txt"
HARD_LIST = DATASET_ROOT / "recall_hard_negatives.txt"
RANDOM_LIST = DATASET_ROOT / "recall_random_negatives.txt"
BASE_MODEL = (
    ROOT
    / "experiments"
    / "segmentation"
    / "training"
    / "mixed_hard_negative"
    / "weights"
    / "best.pt"
)
TRAINING_DIR = ROOT / "experiments" / "segmentation" / "training"
EVALUATION_DIR = ROOT / "experiments" / "segmentation" / "evaluation"
SMOKE_DIR = ROOT / "experiments" / "segmentation" / "smoke-tests"


def select_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recall-oriented YOLO11s-Seg fine-tuning for FracAtlas."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch and validate the saved checkpoint on CPU.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def read_list(path, expected_count):
    if not path.is_file():
        raise FileNotFoundError(f"Missing list: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} entries in {path}, found {len(lines)}."
        )
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"Duplicate entries found in {path}")
    return [Path(line).resolve() for line in lines]


def label_path_for(image_path):
    image_root = (DATASET_ROOT / "images" / "train").resolve()
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as exc:
        raise RuntimeError(f"Image is outside the training split: {image_path}") from exc
    return DATASET_ROOT / "labels" / "train" / relative.with_suffix(".txt")


def check_training_data():
    train_images = read_list(TRAIN_LIST, 855)
    hard_images = read_list(HARD_LIST, 143)
    random_images = read_list(RANDOM_LIST, 142)

    hard_set = set(hard_images)
    random_set = set(random_images)
    if hard_set & random_set:
        raise RuntimeError("Hard-negative and random-negative lists overlap.")

    positive_count = 0
    negative_count = 0
    object_count = 0

    for image_path in train_images:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing training image: {image_path}")
        label_path = label_path_for(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing training label: {label_path}")

        rows = [
            row.strip()
            for row in label_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            if row.strip()
        ]
        if rows:
            positive_count += 1
            object_count += len(rows)
        else:
            negative_count += 1

    if positive_count != 570 or negative_count != 285:
        raise RuntimeError(
            "Expected 570 positives and 285 negatives, "
            f"found {positive_count} positives and {negative_count} negatives."
        )
    if object_count != 757:
        raise RuntimeError(f"Expected 757 positive objects, found {object_count}.")

    negative_set = {path for path in train_images if not label_path_for(path).read_text(
        encoding="utf-8", errors="ignore"
    ).strip()}
    if negative_set != hard_set | random_set:
        raise RuntimeError("The training negatives do not match the two source lists.")

    return {
        "images": len(train_images),
        "positive": positive_count,
        "hard_negative": len(hard_images),
        "random_negative": len(random_images),
        "objects": object_count,
    }


def check_evaluation_data():
    expected = {"val": (480, 91), "test": (340, 69)}
    summary = {}

    for split, (expected_images, expected_objects) in expected.items():
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        images = sorted(image_dir.glob("*.jpg"))
        if len(images) != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} {split} images, found {len(images)}."
            )

        objects = 0
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing {split} label: {label_path}")
            objects += sum(
                bool(row.strip())
                for row in label_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            )

        if objects != expected_objects:
            raise RuntimeError(
                f"Expected {expected_objects} {split} objects, found {objects}."
            )
        summary[split] = {"images": len(images), "objects": objects}

    return summary


def main():
    args = parse_args()
    device = select_device(args.device)

    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Data config not found: {DATA_CONFIG}")
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_MODEL}")

    training_summary = check_training_data()
    evaluation_summary = check_evaluation_data()

    if args.smoke_test:
        epochs = 1
        run_name = "recall_training"
        validation_name = "recall_map_validation"
    else:
        epochs = args.epochs
        run_name = "recall_optimized"
        validation_name = "recall_map_best_validation"

    print(f"Base checkpoint: {BASE_MODEL}")
    print(f"Data config: {DATA_CONFIG}")
    print(f"Training data: {training_summary}")
    print(f"Evaluation data: {evaluation_summary}")
    print(f"Training device: {device}")
    print("Image size: 640")
    print(f"Batch size: {args.batch}")
    print(f"Epoch limit: {epochs}")
    print(f"Run name: {run_name}")

    model = YOLO(str(BASE_MODEL))
    if model.task != "segment":
        raise RuntimeError(f"Expected task='segment', found task={model.task!r}.")

    model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        workers=0,

        seed=91,
        deterministic=True,

        # Low-rate recall-oriented continuation from the mixed-data model.
        optimizer="SGD",
        lr0=0.0005,
        lrf=0.10,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0.5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.005,
        cos_lr=True,
        nbs=64,

        # Disable mAP-based early stopping: every epoch must remain available
        # for the later validation-only recall/F2 checkpoint selection.
        patience=0,
        pretrained=True,
        resume=False,
        cache=False,
        amp=True,
        val=True,

        overlap_mask=True,
        mask_ratio=4,

        # Mild X-ray-safe augmentation; avoid synthetic multi-image compositions.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=2.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,

        project=str(SMOKE_DIR if args.smoke_test else TRAINING_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        # Every epoch is retained so recall-oriented selection can happen later.
        save_period=1,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best_model = Path(model.trainer.best)
    last_model = Path(model.trainer.last)
    results_csv = run_dir / "results.csv"

    if not best_model.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_model}")
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training metrics not found: {results_csv}")

    print("\nRecall-oriented training stage complete.")
    print(f"Ultralytics mAP checkpoint: {best_model}")
    print(f"Last checkpoint: {last_model}")
    print(f"Training metrics: {results_csv}")
    print(f"Run directory: {run_dir}")

    print("\nValidating the saved mAP-best checkpoint on CPU...")

    saved_model = YOLO(str(best_model))
    validation = saved_model.val(
        data=str(DATA_CONFIG),
        split="val",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.70,
        plots=True,
        project=str(SMOKE_DIR if args.smoke_test else EVALUATION_DIR),
        name=validation_name,
        exist_ok=False,
    )

    print("\nSaved mAP-best checkpoint CPU validation metrics:")
    print(validation.results_dict)
    print(f"CPU validation directory: {(SMOKE_DIR if args.smoke_test else EVALUATION_DIR) / validation_name}")

    if args.smoke_test:
        print("\nSmoke test passed. This checkpoint is not a release model.")
    else:
        print(
            "\nTraining finished. Do not use the test split. The next step is "
            "validation-only F2 checkpoint selection from the saved epoch weights."
        )


if __name__ == "__main__":
    main()
