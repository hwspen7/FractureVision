import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "datasets" / "configs" / "current" / "fracatlas_seg_recall_optimized.yaml"
RUN_DIR = ROOT / "experiments" / "segmentation" / "training" / "recall_optimized"
WEIGHTS_DIR = RUN_DIR / "weights"
SELECTION_DIR = ROOT / "experiments" / "segmentation" / "evaluation" / "recall_checkpoint_selection"
BASELINE_WEIGHT = (
    ROOT
    / "experiments"
    / "segmentation"
    / "training"
    / "mixed_hard_negative"
    / "weights"
    / "best.pt"
)
RESULTS_CSV = RUN_DIR / "results.csv"
SCREENING_CSV = SELECTION_DIR / "checkpoint_screening.csv"
OUTPUT_CSV = SELECTION_DIR / "checkpoint_selection.csv"
OUTPUT_JSON = SELECTION_DIR / "checkpoint_selection.json"
OUTPUT_WEIGHT = WEIGHTS_DIR / "f2_best.pt"

# A recall-oriented model is useful only if it keeps an acceptable false-positive
# rate and does not give up too much strict localization quality.
MIN_BOX_PRECISION = 0.50
MIN_MASK_PRECISION = 0.45
MIN_BOX_MAP50_95 = 0.208
MIN_MASK_MAP50_95 = 0.153


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the FracAtlas YOLO11s-Seg checkpoint by validation-only "
            "Box/Mask combined F2, without accessing the test split."
        )
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device. Default: MPS, then CUDA, then CPU.",
    )
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def select_device(requested):
    if requested:
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def check_inputs():
    for path in (DATA_CONFIG, RUN_DIR, WEIGHTS_DIR, BASELINE_WEIGHT, RESULTS_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    rows = list(csv.DictReader(RESULTS_CSV.open(encoding="utf-8-sig")))
    if len(rows) != 20:
        raise RuntimeError(
            f"Expected 20 completed training epochs in {RESULTS_CSV}, found {len(rows)}."
        )

    # Ultralytics epoch0.pt is the checkpoint after displayed epoch 1.
    epoch_weights = sorted(
        WEIGHTS_DIR.glob("epoch*.pt"),
        key=lambda path: int(path.stem.removeprefix("epoch")),
    )
    if len(epoch_weights) != 20:
        raise RuntimeError(
            f"Expected 20 saved epoch checkpoints in {WEIGHTS_DIR}, found {len(epoch_weights)}."
        )

    expected_numbers = list(range(20))
    actual_numbers = [int(path.stem.removeprefix("epoch")) for path in epoch_weights]
    if actual_numbers != expected_numbers:
        raise RuntimeError(
            f"Expected epoch files 0..19, found: {actual_numbers}"
        )

    return epoch_weights


def mean_curve(array):
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return values
    return values.mean(axis=0)


def f2_curve(precision, recall):
    denominator = 4.0 * precision + recall
    return np.divide(
        5.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def validate_checkpoint(weight_path, device, batch, run_name):
    model = YOLO(str(weight_path))
    if model.task != "segment":
        raise RuntimeError(f"Expected a segmentation checkpoint: {weight_path}")

    result = model.val(
        data=str(DATA_CONFIG),
        split="val",
        imgsz=640,
        batch=batch,
        device=device,
        workers=0,
        conf=0.001,
        iou=0.70,
        plots=False,
        project=str(SELECTION_DIR / "validation_runs"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    box_precision = mean_curve(result.box.p_curve)
    box_recall = mean_curve(result.box.r_curve)
    mask_precision = mean_curve(result.seg.p_curve)
    mask_recall = mean_curve(result.seg.r_curve)
    confidence = np.asarray(result.box.px, dtype=np.float64)

    if not (
        confidence.shape
        == box_precision.shape
        == box_recall.shape
        == mask_precision.shape
        == mask_recall.shape
    ):
        raise RuntimeError(f"Metric curve shape mismatch for {weight_path}")

    box_f2 = f2_curve(box_precision, box_recall)
    mask_f2 = f2_curve(mask_precision, mask_recall)
    combined_f2 = (box_f2 + mask_f2) / 2.0

    operating_guard = (
        (box_precision >= MIN_BOX_PRECISION)
        & (mask_precision >= MIN_MASK_PRECISION)
    )
    if operating_guard.any():
        guarded_scores = np.where(operating_guard, combined_f2, -1.0)
        best_index = int(np.argmax(guarded_scores))
        operating_guard_passed = True
    else:
        best_index = int(np.argmax(combined_f2))
        operating_guard_passed = False

    metrics = result.results_dict
    box_map50 = float(metrics["metrics/mAP50(B)"])
    box_map50_95 = float(metrics["metrics/mAP50-95(B)"])
    mask_map50 = float(metrics["metrics/mAP50(M)"])
    mask_map50_95 = float(metrics["metrics/mAP50-95(M)"])
    localization_guard_passed = (
        box_map50_95 >= MIN_BOX_MAP50_95
        and mask_map50_95 >= MIN_MASK_MAP50_95
    )

    return {
        "weight": str(weight_path.resolve()),
        "confidence": float(confidence[best_index]),
        "box_precision": float(box_precision[best_index]),
        "box_recall": float(box_recall[best_index]),
        "box_f2": float(box_f2[best_index]),
        "mask_precision": float(mask_precision[best_index]),
        "mask_recall": float(mask_recall[best_index]),
        "mask_f2": float(mask_f2[best_index]),
        "combined_f2": float(combined_f2[best_index]),
        "box_map50": box_map50,
        "box_map50_95": box_map50_95,
        "mask_map50": mask_map50,
        "mask_map50_95": mask_map50_95,
        "operating_guard_passed": operating_guard_passed,
        "localization_guard_passed": localization_guard_passed,
    }


def save_csv(path, records):
    fieldnames = [
        "candidate",
        "display_epoch",
        "weight",
        "confidence",
        "box_precision",
        "box_recall",
        "box_f2",
        "mask_precision",
        "mask_recall",
        "mask_f2",
        "combined_f2",
        "box_map50",
        "box_map50_95",
        "mask_map50",
        "mask_map50_95",
        "operating_guard_passed",
        "localization_guard_passed",
        "eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    device = select_device(args.device)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    epoch_weights = check_inputs()

    candidates = [("baseline_final_v1", None, BASELINE_WEIGHT)]
    candidates.extend(
        (
            f"recall_optimized_epoch_{index + 1:02d}",
            index + 1,
            weight_path,
        )
        for index, weight_path in enumerate(epoch_weights)
    )

    print(f"Validation device: {device}")
    print(f"Candidates: {len(candidates)} (1 baseline + 20 recall-v2 epochs)")
    print("Split: val only; test is never accessed")
    print(
        "Guardrails: "
        f"Box P >= {MIN_BOX_PRECISION:.2f}, "
        f"Mask P >= {MIN_MASK_PRECISION:.2f}, "
        f"Box mAP50-95 >= {MIN_BOX_MAP50_95:.3f}, "
        f"Mask mAP50-95 >= {MIN_MASK_MAP50_95:.3f}"
    )

    screening_records = []
    for number, (candidate, display_epoch, weight_path) in enumerate(candidates, 1):
        print(f"\n[{number}/{len(candidates)}] {candidate}")
        metrics = validate_checkpoint(
            weight_path=weight_path,
            device=device,
            batch=args.batch,
            run_name=candidate,
        )
        record = {
            "candidate": candidate,
            "display_epoch": "" if display_epoch is None else display_epoch,
            **metrics,
        }
        record["eligible"] = bool(
            record["operating_guard_passed"]
            and record["localization_guard_passed"]
        )
        screening_records.append(record)
        print(
            f"conf={record['confidence']:.3f} | "
            f"Box P/R/F2={record['box_precision']:.3f}/"
            f"{record['box_recall']:.3f}/{record['box_f2']:.3f} | "
            f"Mask P/R/F2={record['mask_precision']:.3f}/"
            f"{record['mask_recall']:.3f}/{record['mask_f2']:.3f} | "
            f"combined F2={record['combined_f2']:.3f} | "
            f"eligible={record['eligible']}"
        )

    save_csv(SCREENING_CSV, screening_records)
    eligible_recall = [
        record
        for record in screening_records
        if record["eligible"] and record["candidate"] != "baseline_final_v1"
    ]
    if not eligible_recall:
        raise RuntimeError(
            "No recall-v2 checkpoint passed every screening guardrail. "
            "No f2_best.pt was created. "
            f"Inspect: {SCREENING_CSV}"
        )

    # MPS/CUDA makes the full 21-checkpoint scan practical. The baseline and
    # three best recall-v2 candidates are then re-evaluated on CPU, and the
    # final decision is made only from those CPU results.
    top_recall = sorted(
        eligible_recall,
        key=lambda record: record["combined_f2"],
        reverse=True,
    )[:3]
    confirmation_candidates = [
        next(
            record
            for record in screening_records
            if record["candidate"] == "baseline_final_v1"
        ),
        *top_recall,
    ]

    print("\nCPU confirmation of baseline and top recall-v2 candidates...")
    confirmed_records = []
    for number, screened in enumerate(confirmation_candidates, 1):
        print(
            f"\n[CPU {number}/{len(confirmation_candidates)}] "
            f"{screened['candidate']}"
        )
        metrics = validate_checkpoint(
            weight_path=Path(screened["weight"]),
            device="cpu",
            batch=args.batch,
            run_name=f"cpu_{screened['candidate']}",
        )
        record = {
            "candidate": screened["candidate"],
            "display_epoch": screened["display_epoch"],
            **metrics,
        }
        record["eligible"] = bool(
            record["operating_guard_passed"]
            and record["localization_guard_passed"]
        )
        confirmed_records.append(record)
        print(
            f"conf={record['confidence']:.3f} | "
            f"Box P/R/F2={record['box_precision']:.3f}/"
            f"{record['box_recall']:.3f}/{record['box_f2']:.3f} | "
            f"Mask P/R/F2={record['mask_precision']:.3f}/"
            f"{record['mask_recall']:.3f}/{record['mask_f2']:.3f} | "
            f"combined F2={record['combined_f2']:.3f} | "
            f"eligible={record['eligible']}"
        )

    save_csv(OUTPUT_CSV, confirmed_records)
    confirmed_eligible = [record for record in confirmed_records if record["eligible"]]
    if not confirmed_eligible:
        raise RuntimeError(
            "No CPU-confirmed checkpoint passed every guardrail. "
            "No f2_best.pt was created. "
            f"Inspect: {OUTPUT_CSV}"
        )

    winner = max(confirmed_eligible, key=lambda record: record["combined_f2"])
    baseline = next(
        record
        for record in confirmed_records
        if record["candidate"] == "baseline_final_v1"
    )
    winner["combined_f2_gain_over_baseline"] = (
        winner["combined_f2"] - baseline["combined_f2"]
    )

    shutil.copy2(winner["weight"], OUTPUT_WEIGHT)

    report = {
        "selection_basis": (
            "Highest validation Box/Mask mean F2 among checkpoints passing "
            "predeclared precision and mAP50-95 guardrails."
        ),
        "test_split_accessed": False,
        "guardrails": {
            "minimum_box_precision": MIN_BOX_PRECISION,
            "minimum_mask_precision": MIN_MASK_PRECISION,
            "minimum_box_map50_95": MIN_BOX_MAP50_95,
            "minimum_mask_map50_95": MIN_MASK_MAP50_95,
        },
        "baseline": baseline,
        "selected": winner,
        "saved_weight": str(OUTPUT_WEIGHT.resolve()),
        "screening_results_csv": str(SCREENING_CSV.resolve()),
        "all_results_csv": str(OUTPUT_CSV.resolve()),
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSelection complete.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSelected checkpoint copy: {OUTPUT_WEIGHT}")
    print(f"Full screening CSV: {SCREENING_CSV}")
    print(f"CPU confirmation CSV: {OUTPUT_CSV}")
    print(f"Selection report: {OUTPUT_JSON}")
    print("Test split was not accessed.")


if __name__ == "__main__":
    main()
