import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "datasets" / "configs" / "current" / "fracatlas_seg_positive.yaml"
SEGMENT_DATASET = ROOT / "datasets" / "fracatlas_seg_yolo"
PRETRAINED_MODEL_NAME = "yolo11s-seg.pt"
PROJECT_DIR = ROOT / "runs" / "segment"

EXPECTED_IMAGES = {
    "train": 570,
    "val": 82,
    "test": 63,
}


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
        description="Train the positive-only YOLO11s segmentation benchmark."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch and validate the saved checkpoint on CPU.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device such as 0, mps, or cpu; defaults to automatic selection.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


def check_positive_lists():
    counts = {}

    for split, expected_count in EXPECTED_IMAGES.items():
        list_path = SEGMENT_DATASET / f"{split}_positive.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing positive list: {list_path}")

        paths = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(paths) != expected_count:
            raise RuntimeError(
                f"{split} should contain {expected_count} images, found {len(paths)}."
            )
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Duplicate image paths found in {list_path}")

        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing images referenced by {list_path}: {missing[:5]}"
            )

        counts[split] = len(paths)

    split_paths = {
        split: set(
            line.strip()
            for line in (SEGMENT_DATASET / f"{split}_positive.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        for split in EXPECTED_IMAGES
    }

    if split_paths["train"] & split_paths["val"]:
        raise RuntimeError("Train/validation positive lists overlap.")
    if split_paths["train"] & split_paths["test"]:
        raise RuntimeError("Train/test positive lists overlap.")
    if split_paths["val"] & split_paths["test"]:
        raise RuntimeError("Validation/test positive lists overlap.")

    return counts


def main():
    args = parse_args()
    device = select_device(args.device)

    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Data config not found: {DATA_CONFIG}")

    split_counts = check_positive_lists()

    if args.smoke_test:
        epochs = 1
        run_name = "fracatlas_yolo11s_seg_positive_smoke_v2"
    else:
        epochs = args.epochs
        run_name = "fracatlas_yolo11s_seg_positive_v2"

    print(f"Pretrained model: {PRETRAINED_MODEL_NAME}")
    print(f"Data config: {DATA_CONFIG}")
    print(f"Positive split sizes: {split_counts}")
    print(f"Training device: {device}")
    print("Image size: 640")
    print(f"Batch size: {args.batch}")
    print(f"Epoch limit: {epochs}")
    print(f"Run name: {run_name}")

    model = YOLO(PRETRAINED_MODEL_NAME)
    if model.task != "segment":
        raise RuntimeError(
            f"Expected a segmentation model, but loaded task={model.task!r}."
        )

    model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        workers=0,

        # Reproducibility
        seed=51,
        deterministic=True,

        # Official FracAtlas release-weight recipe, with YOLO11s-Seg
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=False,
        nbs=64,

        # Training control
        patience=25,
        pretrained=True,
        resume=False,
        cache=False,
        amp=True,

        # Match the released FracAtlas segmentation checkpoint
        overlap_mask=True,
        mask_ratio=4,

        # Official standard augmentation values
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        # Laterality is not a target class; match the official flip setting
        fliplr=0.5,
        flipud=0.0,

        # The strongest released official weight records mosaic disabled
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,

        # Outputs
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        save_period=5,
        val=True,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best_model = Path(model.trainer.best)
    last_model = Path(model.trainer.last)
    results_csv = run_dir / "results.csv"

    if not best_model.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_model}")

    print("\nTraining stage complete.")
    print(f"Best checkpoint: {best_model}")
    print(f"Last checkpoint: {last_model}")
    print(f"Training metrics: {results_csv}")
    print(f"Run directory: {run_dir}")

    validation_name = f"{run_name}_best_cpu_val"
    print("\nValidating the saved best checkpoint on CPU...")

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
        project=str(PROJECT_DIR),
        name=validation_name,
        exist_ok=False,
    )

    print("\nSaved-checkpoint CPU validation metrics:")
    print(validation.results_dict)
    print(f"CPU validation directory: {PROJECT_DIR / validation_name}")

    if args.smoke_test:
        print("\nSmoke test passed. This checkpoint is not a release model.")
    else:
        print(
            "\nPositive benchmark training finished. Do not evaluate the test split "
            "until model selection and negative calibration are complete."
        )


if __name__ == "__main__":
    main()
