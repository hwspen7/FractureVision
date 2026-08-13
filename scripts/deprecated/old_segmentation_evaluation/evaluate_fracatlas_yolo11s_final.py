import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "datasets").is_dir():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = Path.cwd()

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "fracatlas_yolo11s_seg_final_v1"
    / "weights"
    / "best.pt"
)
DATA_CONFIG = PROJECT_ROOT / "datasets" / "fracatlas_seg_final.yaml"
PROJECT_DIR = PROJECT_ROOT / "runs" / "segment"

# These are reporting points, not hiding rules for the future Qt interface.
REPORT_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the locked YOLO11s-Seg model on the FracAtlas test split."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def metric_at_threshold(metric, threshold):
    index = int(np.argmin(np.abs(metric.px - threshold)))
    return {
        "threshold": float(metric.px[index]),
        "precision": float(metric.p_curve.mean(axis=0)[index]),
        "recall": float(metric.r_curve.mean(axis=0)[index]),
        "f1": float(metric.f1_curve.mean(axis=0)[index]),
    }


def print_operating_point(label, values):
    print(
        f"{label:<5} "
        f"P={values['precision']:.4f} "
        f"R={values['recall']:.4f} "
        f"F1={values['f1']:.4f}"
    )


def main():
    args = parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Data config not found: {DATA_CONFIG}")

    print(f"Locked model: {MODEL_PATH}")
    print(f"Test data config: {DATA_CONFIG}")
    print(f"Device: {args.device}")
    print("Evaluation confidence floor: 0.001")
    print("The test split is evaluated once; do not tune the model from this result.\n")

    model = YOLO(str(MODEL_PATH))
    if model.task != "segment":
        raise RuntimeError(f"Expected task='segment', found task={model.task!r}.")

    result = model.val(
        data=str(DATA_CONFIG),
        split="test",
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.70,
        plots=True,
        project=str(PROJECT_DIR),
        name="fracatlas_yolo11s_seg_final_v1_locked_test",
        exist_ok=False,
    )

    print("\nStandard threshold-independent test metrics:")
    for key, value in result.results_dict.items():
        print(f"{key}: {float(value):.6f}")

    print("\nFixed confidence operating points:")
    for threshold in REPORT_THRESHOLDS:
        print(f"\nConfidence {threshold:.0%}")
        print_operating_point(
            "Box",
            metric_at_threshold(result.box, threshold),
        )
        print_operating_point(
            "Mask",
            metric_at_threshold(result.seg, threshold),
        )

    print(f"\nResults directory: {result.save_dir}")
    print(
        "Test evaluation complete. Keep every confidence result for reporting, "
        "and do not retrain or select another checkpoint from the test metrics."
    )


if __name__ == "__main__":
    main()
