import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "datasets" / "configs" / "current" / "fracatlas_seg_final.yaml"
SEGMENT_DATASET = ROOT / "datasets" / "fracatlas_seg_yolo"
TRAIN_LIST = SEGMENT_DATASET / "train_final_mixed.txt"
BASE_MODEL = (
    ROOT
    / "runs"
    / "segment"
    / "fracatlas_yolo11s_seg_positive_v2"
    / "weights"
    / "best.pt"
)
PROJECT_DIR = ROOT / "runs" / "segment"


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
        description="Fine-tune the final FracAtlas segmentation model with hard negatives."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch and validate the saved checkpoint on CPU.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args()


def label_path_for(image_path):
    image_root = SEGMENT_DATASET / "images" / "train"
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as exc:
        raise RuntimeError(f"Training image is outside {image_root}: {image_path}") from exc
    return SEGMENT_DATASET / "labels" / "train" / relative.with_suffix(".txt")


def check_training_list():
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"Training list not found: {TRAIN_LIST}")

    lines = [
        line.strip()
        for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1140:
        raise RuntimeError(f"Expected 1140 training images, found {len(lines)}.")
    if len(lines) != len(set(lines)):
        raise RuntimeError("Duplicate paths found in the final training list.")

    positive_count = 0
    negative_count = 0
    object_count = 0

    for line in lines:
        image_path = Path(line).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Training image not found: {image_path}")

        label_path = label_path_for(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Training label not found: {label_path}")

        labels = [
            row.strip()
            for row in label_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            if row.strip()
        ]
        if labels:
            positive_count += 1
            object_count += len(labels)
        else:
            negative_count += 1

    if positive_count != 570 or negative_count != 570:
        raise RuntimeError(
            "Expected 570 positive and 570 negative training images, "
            f"found {positive_count} positive and {negative_count} negative."
        )
    if object_count != 757:
        raise RuntimeError(f"Expected 757 positive objects, found {object_count}.")

    return {
        "images": len(lines),
        "positive": positive_count,
        "hard_negative": negative_count,
        "objects": object_count,
    }


def check_validation_split():
    expected = {"val": (480, 91), "test": (340, 69)}
    summary = {}

    for split, (expected_images, expected_objects) in expected.items():
        image_dir = SEGMENT_DATASET / "images" / split
        label_dir = SEGMENT_DATASET / "labels" / split
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
                bool(line.strip())
                for line in label_path.read_text(
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

    train_summary = check_training_list()
    evaluation_summary = check_validation_split()

    if args.smoke_test:
        epochs = 1
        run_name = "fracatlas_yolo11s_seg_final_smoke_v1"
    else:
        epochs = args.epochs
        run_name = "fracatlas_yolo11s_seg_final_v1"

    print(f"Base checkpoint: {BASE_MODEL}")
    print(f"Data config: {DATA_CONFIG}")
    print(f"Training data: {train_summary}")
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

        # Reproducible hard-negative fine-tuning.
        seed=71,
        deterministic=True,

        # Preserve learned fracture localization while adapting confidence.
        optimizer="SGD",
        lr0=0.002,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        nbs=64,

        # Select best.pt on the complete mixed validation split.
        patience=15,
        pretrained=True,
        resume=False,
        cache=False,
        amp=True,
        val=True,

        overlap_mask=True,
        mask_ratio=4,

        # Mild X-ray-safe augmentation for fine-tuning.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.20,
        degrees=2.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,

        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        save_period=5,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best_model = Path(model.trainer.best)
    last_model = Path(model.trainer.last)
    results_csv = run_dir / "results.csv"

    if not best_model.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_model}")

    print("\nFine-tuning stage complete.")
    print(f"Best checkpoint: {best_model}")
    print(f"Last checkpoint: {last_model}")
    print(f"Training metrics: {results_csv}")
    print(f"Run directory: {run_dir}")

    validation_name = f"{run_dir.name}_best_cpu_val"
    print("\nValidating the saved best checkpoint on the complete validation split...")

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
            "\nFinal mixed fine-tuning finished. Do not evaluate the test split "
            "until this model is compared with the positive-only checkpoint."
        )


if __name__ == "__main__":
    main()
