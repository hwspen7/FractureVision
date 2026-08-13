import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou


ROOT = Path(
    os.environ.get(
        "FRACATLAS_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).resolve()

DEFAULT_MODEL = (
    ROOT
    / "runs"
    / "segment"
    / "fracatlas_yolo11s_seg_recall_v2"
    / "weights"
    / "f2_best.pt"
)
DATA_CONFIG = ROOT / "datasets" / "configs" / "current" / "fracatlas_seg_recall_v2.yaml"
DATASET_ROOT = ROOT / "datasets" / "fracatlas_seg_yolo"
VAL_IMAGE_DIR = DATASET_ROOT / "images" / "val"
VAL_LABEL_DIR = DATASET_ROOT / "labels" / "val"
PROJECT_DIR = Path(
    os.environ.get(
        "FRACATLAS_OUTPUT_ROOT",
        ROOT / "runs" / "segment",
    )
).resolve()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)
MIN_BOX_PRECISION = 0.50
MIN_MASK_PRECISION = 0.45
MIN_MEAN_MAP50_GAIN = 0.02
MAX_MEAN_MAP50_95_DROP = 0.01
MAX_RECALL_DROP = 0.005
MAX_F2_DROP = 0.002
PARITY_MAP50_TOLERANCE = 0.03
PARITY_MAP50_95_TOLERANCE = 0.03

VARIANTS = {
    "original": (0,),
    "original_plus_flip": (0, 1),
    "original_plus_flip_plus_contrast": (0, 1, 2),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only single-model TTA evaluation for FracAtlas "
            "YOLO11s-Seg. The test split is never accessed."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--fusion-iou", type=float, default=0.45)
    parser.add_argument(
        "--name",
        default="fracatlas_yolo11s_seg_recall_v2_tta_val_v1",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 8 positive and 8 negative validation images only.",
    )
    return parser.parse_args()


def select_device(requested):
    if requested:
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def check_inputs(model_path):
    required = (
        model_path,
        DATA_CONFIG,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def read_image(path):
    with Image.open(path) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mild_contrast(image):
    """Apply a deliberately mild, geometry-preserving X-ray contrast TTA."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    return cv2.addWeighted(image, 0.50, enhanced, 0.50, 0.0)


def build_views(image):
    return [
        image,
        np.ascontiguousarray(image[:, ::-1]),
        mild_contrast(image),
    ]


def box_iou_numpy(box_a, box_b):
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0,
        float(box_a[3] - box_a[1]),
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0,
        float(box_b[3] - box_b[1]),
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def result_to_predictions(result, view_id, image_width):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or result.masks.data is None:
        raise RuntimeError(
            "The checkpoint produced boxes without masks. "
            "A YOLO segmentation checkpoint is required."
        )

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    masks = result.masks.data.detach().cpu().numpy().astype(np.float32)

    if len(boxes) != len(masks):
        raise RuntimeError("Prediction box/mask count mismatch.")

    predictions = []
    for box, confidence, class_id, prediction_mask in zip(
        boxes,
        confidences,
        classes,
        masks,
    ):
        if int(class_id) != 0:
            raise RuntimeError(f"Unexpected predicted class: {class_id}")

        restored_box = box.copy()
        restored_mask = prediction_mask
        if view_id == 1:
            x1 = float(restored_box[0])
            x2 = float(restored_box[2])
            restored_box[0] = image_width - x2
            restored_box[2] = image_width - x1
            restored_mask = np.ascontiguousarray(restored_mask[:, ::-1])

        predictions.append(
            {
                "box": restored_box,
                "confidence": float(confidence),
                "class_id": 0,
                "mask": restored_mask,
                "view_id": view_id,
            }
        )
    return predictions


def weighted_cluster_box(members):
    weights = np.asarray(
        [member["confidence"] for member in members],
        dtype=np.float32,
    )
    boxes = np.stack(
        [member["box"] for member in members],
        axis=0,
    )
    return np.average(boxes, axis=0, weights=weights).astype(np.float32)


def fuse_predictions(predictions_by_view, selected_views, fusion_iou):
    """
    Fuse geometrically aligned predictions from one model.

    Confidence follows the standard weighted-box-fusion idea: the sum of
    member confidences is divided by the number of enabled views. A detection
    therefore gains rank only when the same location is supported by multiple
    views; a one-view false alarm is penalized instead of being amplified.
    """
    selected = []
    for view_id in selected_views:
        selected.extend(predictions_by_view[view_id])

    selected.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )
    clusters = []

    for prediction in selected:
        best_cluster = None
        best_iou = fusion_iou
        for cluster in clusters:
            used_views = {
                member["view_id"]
                for member in cluster["members"]
            }
            if prediction["view_id"] in used_views:
                continue
            overlap = box_iou_numpy(
                prediction["box"],
                cluster["box"],
            )
            if overlap >= best_iou:
                best_iou = overlap
                best_cluster = cluster

        if best_cluster is None:
            clusters.append(
                {
                    "members": [prediction],
                    "box": prediction["box"].copy(),
                }
            )
        else:
            best_cluster["members"].append(prediction)
            best_cluster["box"] = weighted_cluster_box(
                best_cluster["members"]
            )

    fused = []
    enabled_view_count = len(selected_views)
    for cluster in clusters:
        members = cluster["members"]
        weights = np.asarray(
            [member["confidence"] for member in members],
            dtype=np.float32,
        )
        mask_stack = np.stack(
            [member["mask"] for member in members],
            axis=0,
        )
        fused_mask = np.average(
            mask_stack,
            axis=0,
            weights=weights,
        )
        fused_confidence = float(weights.sum() / enabled_view_count)
        fused.append(
            {
                "box": weighted_cluster_box(members),
                "confidence": min(fused_confidence, 1.0),
                "class_id": 0,
                "mask": fused_mask >= 0.50,
                "support": len(members),
            }
        )

    fused.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )
    return fused


def letterbox_geometry(image_width, image_height, mask_width, mask_height):
    gain = min(
        mask_width / image_width,
        mask_height / image_height,
    )
    resized_width = round(image_width * gain)
    resized_height = round(image_height * gain)
    pad_x = (mask_width - resized_width) / 2.0
    pad_y = (mask_height - resized_height) / 2.0
    return gain, pad_x, pad_y


def read_ground_truth(
    label_path,
    image_width,
    image_height,
    mask_width,
    mask_height,
):
    if not label_path.is_file():
        raise FileNotFoundError(f"Validation label not found: {label_path}")

    boxes = []
    masks = []
    gain, pad_x, pad_y = letterbox_geometry(
        image_width,
        image_height,
        mask_width,
        mask_height,
    )

    for line_number, line in enumerate(
        label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(),
        start=1,
    ):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(
                f"Invalid segmentation row: {label_path}:{line_number}"
            )

        class_id = int(float(parts[0]))
        if class_id != 0:
            raise ValueError(
                f"Unexpected class {class_id}: {label_path}:{line_number}"
            )

        normalized = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        original_points = normalized.copy()
        original_points[:, 0] *= image_width
        original_points[:, 1] *= image_height
        boxes.append(
            [
                float(original_points[:, 0].min()),
                float(original_points[:, 1].min()),
                float(original_points[:, 0].max()),
                float(original_points[:, 1].max()),
            ]
        )

        mask_points = original_points.copy()
        mask_points[:, 0] = mask_points[:, 0] * gain + pad_x
        mask_points[:, 1] = mask_points[:, 1] * gain + pad_y
        mask_points[:, 0] = np.clip(mask_points[:, 0], 0, mask_width - 1)
        mask_points[:, 1] = np.clip(mask_points[:, 1], 0, mask_height - 1)

        target_mask = np.zeros(
            (mask_height, mask_width),
            dtype=np.uint8,
        )
        cv2.fillPoly(
            target_mask,
            [np.rint(mask_points).astype(np.int32)],
            1,
        )
        masks.append(target_mask.astype(bool))

    boxes_array = (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if boxes
        else np.zeros((0, 4), dtype=np.float32)
    )
    masks_array = (
        np.stack(masks, axis=0)
        if masks
        else np.zeros(
            (0, mask_height, mask_width),
            dtype=bool,
        )
    )
    return boxes_array, masks_array


def match_iou(predicted_classes, true_classes, iou):
    correct = np.zeros(
        (len(predicted_classes), len(IOU_THRESHOLDS)),
        dtype=bool,
    )
    if len(predicted_classes) == 0 or len(true_classes) == 0:
        return correct

    correct_class = true_classes[:, None] == predicted_classes[None, :]
    filtered_iou = iou * correct_class

    for threshold_index, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.nonzero(filtered_iou >= threshold)
        matches = np.asarray(matches).T
        if not len(matches):
            continue
        if len(matches) > 1:
            match_ious = filtered_iou[
                matches[:, 0],
                matches[:, 1],
            ]
            matches = matches[match_ious.argsort()[::-1]]
            matches = matches[
                np.unique(matches[:, 1], return_index=True)[1]
            ]
            matches = matches[
                np.unique(matches[:, 0], return_index=True)[1]
            ]
        correct[
            matches[:, 1].astype(int),
            threshold_index,
        ] = True
    return correct


def create_stats():
    return {
        "tp": [],
        "tp_m": [],
        "conf": [],
        "pred_cls": [],
        "target_cls": [],
    }


def append_stats(stats, predictions, true_boxes, true_masks):
    if predictions:
        predicted_boxes = np.stack(
            [prediction["box"] for prediction in predictions],
            axis=0,
        ).astype(np.float32)
        predicted_masks = np.stack(
            [prediction["mask"] for prediction in predictions],
            axis=0,
        ).astype(bool)
        confidences = np.asarray(
            [prediction["confidence"] for prediction in predictions],
            dtype=np.float32,
        )
        predicted_classes = np.zeros(
            (len(predictions),),
            dtype=np.float32,
        )
    else:
        predicted_boxes = np.zeros((0, 4), dtype=np.float32)
        predicted_masks = np.zeros(
            (0, *true_masks.shape[1:]),
            dtype=bool,
        )
        confidences = np.zeros((0,), dtype=np.float32)
        predicted_classes = np.zeros((0,), dtype=np.float32)

    true_classes = np.zeros((len(true_boxes),), dtype=np.float32)

    if len(predicted_boxes) and len(true_boxes):
        box_overlaps = box_iou(
            torch.from_numpy(true_boxes),
            torch.from_numpy(predicted_boxes),
        ).numpy()
        mask_overlaps = mask_iou(
            torch.from_numpy(true_masks.reshape(len(true_masks), -1)).float(),
            torch.from_numpy(
                predicted_masks.reshape(len(predicted_masks), -1)
            ).float(),
        ).numpy()
    else:
        box_overlaps = np.zeros(
            (len(true_boxes), len(predicted_boxes)),
            dtype=np.float32,
        )
        mask_overlaps = np.zeros_like(box_overlaps)

    stats["tp"].append(
        match_iou(
            predicted_classes,
            true_classes,
            box_overlaps,
        )
    )
    stats["tp_m"].append(
        match_iou(
            predicted_classes,
            true_classes,
            mask_overlaps,
        )
    )
    stats["conf"].append(confidences)
    stats["pred_cls"].append(predicted_classes)
    stats["target_cls"].append(true_classes)


def mean_curve(values):
    array = np.asarray(values, dtype=np.float64)
    return array if array.ndim == 1 else array.mean(axis=0)


def f2_curve(precision, recall):
    denominator = 4.0 * precision + recall
    return np.divide(
        5.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def calculate_metrics(stats, output_dir, make_plots):
    output_dir.mkdir(parents=True, exist_ok=False)
    combined = {
        key: np.concatenate(values, axis=0)
        for key, values in stats.items()
    }

    metrics = SegmentMetrics(
        save_dir=output_dir,
        plot=make_plots,
        names={0: "fracture"},
    )
    metrics.process(
        tp=combined["tp"],
        tp_m=combined["tp_m"],
        conf=combined["conf"],
        pred_cls=combined["pred_cls"],
        target_cls=combined["target_cls"],
    )

    box_precision = mean_curve(metrics.box.p_curve)
    box_recall = mean_curve(metrics.box.r_curve)
    mask_precision = mean_curve(metrics.seg.p_curve)
    mask_recall = mean_curve(metrics.seg.r_curve)
    confidence = np.asarray(metrics.box.px, dtype=np.float64)

    box_f2 = f2_curve(box_precision, box_recall)
    mask_f2 = f2_curve(mask_precision, mask_recall)
    combined_f2 = (box_f2 + mask_f2) / 2.0
    guard = (
        (box_precision >= MIN_BOX_PRECISION)
        & (mask_precision >= MIN_MASK_PRECISION)
    )
    if guard.any():
        best_index = int(np.argmax(np.where(guard, combined_f2, -1.0)))
        precision_guard_passed = True
    else:
        best_index = int(np.argmax(combined_f2))
        precision_guard_passed = False

    standard = {
        key: float(value)
        for key, value in metrics.results_dict.items()
    }
    operating_point = {
        "confidence": float(confidence[best_index]),
        "box_precision": float(box_precision[best_index]),
        "box_recall": float(box_recall[best_index]),
        "box_f2": float(box_f2[best_index]),
        "mask_precision": float(mask_precision[best_index]),
        "mask_recall": float(mask_recall[best_index]),
        "mask_f2": float(mask_f2[best_index]),
        "combined_f2": float(combined_f2[best_index]),
        "mean_recall": float(
            (box_recall[best_index] + mask_recall[best_index]) / 2.0
        ),
        "precision_guard_passed": precision_guard_passed,
    }
    return {
        "standard_metrics": standard,
        "f2_operating_point": operating_point,
    }


def choose_smoke_images(image_paths):
    positive = []
    negative = []
    for image_path in image_paths:
        label_path = VAL_LABEL_DIR / f"{image_path.stem}.txt"
        is_positive = bool(
            label_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        )
        (positive if is_positive else negative).append(image_path)
    if len(positive) < 8 or len(negative) < 8:
        raise RuntimeError("Not enough positive/negative images for smoke test.")
    return positive[:8] + negative[:8]


def metric_value(record, key):
    return record["standard_metrics"][key]


def add_comparison_fields(record, baseline):
    record["mean_map50"] = (
        metric_value(record, "metrics/mAP50(B)")
        + metric_value(record, "metrics/mAP50(M)")
    ) / 2.0
    record["mean_map50_95"] = (
        metric_value(record, "metrics/mAP50-95(B)")
        + metric_value(record, "metrics/mAP50-95(M)")
    ) / 2.0

    if baseline is None:
        record["mean_map50_gain"] = 0.0
        record["mean_map50_95_gain"] = 0.0
        record["mean_recall_gain"] = 0.0
        record["combined_f2_gain"] = 0.0
        return

    record["mean_map50_gain"] = (
        record["mean_map50"] - baseline["mean_map50"]
    )
    record["mean_map50_95_gain"] = (
        record["mean_map50_95"] - baseline["mean_map50_95"]
    )
    record["mean_recall_gain"] = (
        record["f2_operating_point"]["mean_recall"]
        - baseline["f2_operating_point"]["mean_recall"]
    )
    record["combined_f2_gain"] = (
        record["f2_operating_point"]["combined_f2"]
        - baseline["f2_operating_point"]["combined_f2"]
    )


def approval_decision(record, metric_parity_passed):
    return bool(
        metric_parity_passed
        and record["mean_map50_gain"] >= MIN_MEAN_MAP50_GAIN
        and record["mean_map50_95_gain"] >= -MAX_MEAN_MAP50_95_DROP
        and record["mean_recall_gain"] >= -MAX_RECALL_DROP
        and record["combined_f2_gain"] >= -MAX_F2_DROP
        and record["f2_operating_point"]["precision_guard_passed"]
    )


def builtin_baseline_validation(
    model,
    model_path,
    device,
    batch,
    imgsz,
    output_dir,
):
    result = model.val(
        data=str(DATA_CONFIG),
        split="val",
        imgsz=imgsz,
        batch=max(batch, 1),
        device=device,
        workers=0,
        conf=0.001,
        iou=0.70,
        plots=False,
        project=str(output_dir),
        name="builtin_baseline",
        exist_ok=False,
        verbose=False,
    )
    return {
        "model": str(model_path),
        **{
            key: float(value)
            for key, value in result.results_dict.items()
        },
    }


def parity_check(custom_baseline, builtin_baseline):
    differences = {}
    passed = True
    for suffix in ("B", "M"):
        key_50 = f"metrics/mAP50({suffix})"
        key_95 = f"metrics/mAP50-95({suffix})"
        difference_50 = abs(
            metric_value(custom_baseline, key_50)
            - builtin_baseline[key_50]
        )
        difference_95 = abs(
            metric_value(custom_baseline, key_95)
            - builtin_baseline[key_95]
        )
        differences[key_50] = difference_50
        differences[key_95] = difference_95
        passed = bool(
            passed
            and difference_50 <= PARITY_MAP50_TOLERANCE
            and difference_95 <= PARITY_MAP50_95_TOLERANCE
        )
    return passed, differences


def save_comparison_csv(path, records):
    rows = []
    for name, record in records.items():
        standard = record["standard_metrics"]
        operating = record["f2_operating_point"]
        rows.append(
            {
                "variant": name,
                "box_precision": standard["metrics/precision(B)"],
                "box_recall": standard["metrics/recall(B)"],
                "box_map50": standard["metrics/mAP50(B)"],
                "box_map50_95": standard["metrics/mAP50-95(B)"],
                "mask_precision": standard["metrics/precision(M)"],
                "mask_recall": standard["metrics/recall(M)"],
                "mask_map50": standard["metrics/mAP50(M)"],
                "mask_map50_95": standard["metrics/mAP50-95(M)"],
                "f2_confidence": operating["confidence"],
                "f2_mean_recall": operating["mean_recall"],
                "combined_f2": operating["combined_f2"],
                "mean_map50_gain": record["mean_map50_gain"],
                "mean_map50_95_gain": record["mean_map50_95_gain"],
                "mean_recall_gain": record["mean_recall_gain"],
                "combined_f2_gain": record["combined_f2_gain"],
                "approved": record.get("approved", False),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not 0.20 <= args.fusion_iou <= 0.80:
        raise ValueError("--fusion-iou must be between 0.20 and 0.80.")

    model_path = args.model.resolve()
    check_inputs(model_path)
    device = select_device(args.device)

    image_paths = sorted(
        path
        for path in VAL_IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) != 480:
        raise RuntimeError(
            f"Expected 480 validation images, found {len(image_paths)}."
        )

    smoke_test = bool(args.smoke_test)
    if smoke_test:
        image_paths = choose_smoke_images(image_paths)
        output_name = (
            args.name
            if args.name.endswith("_smoke")
            else f"{args.name}_smoke"
        )
    else:
        output_name = args.name

    output_dir = PROJECT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Validation images: {len(image_paths)}")
    print(f"Image size: {args.imgsz}")
    print(f"Fusion IoU: {args.fusion_iou}")
    print(f"Smoke test: {smoke_test}")
    print("Views: original, horizontal flip, mild CLAHE contrast")
    print("Test split accessed: False")

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise RuntimeError("A YOLO segmentation checkpoint is required.")

    stats_by_variant = {
        name: create_stats()
        for name in VARIANTS
    }
    count_rows = []
    start_time = time.time()

    for image_index, image_path in enumerate(image_paths, start=1):
        image = read_image(image_path)
        image_height, image_width = image.shape[:2]
        views = build_views(image)

        results = model.predict(
            source=views,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            conf=0.001,
            iou=0.70,
            max_det=100,
            retina_masks=False,
            save=False,
            verbose=False,
        )
        if len(results) != 3:
            raise RuntimeError(
                f"Expected 3 TTA results for {image_path.name}, got {len(results)}."
            )

        predictions_by_view = {
            view_id: result_to_predictions(
                result,
                view_id=view_id,
                image_width=image_width,
            )
            for view_id, result in enumerate(results)
        }

        mask_shape = next(
            (
                tuple(result.masks.data.shape[-2:])
                for result in results
                if result.masks is not None
                and result.masks.data is not None
            ),
            tuple(results[0].orig_shape),
        )
        mask_height, mask_width = mask_shape

        true_boxes, true_masks = read_ground_truth(
            VAL_LABEL_DIR / f"{image_path.stem}.txt",
            image_width=image_width,
            image_height=image_height,
            mask_width=int(mask_width),
            mask_height=int(mask_height),
        )

        row = {
            "image": image_path.name,
            "ground_truth_objects": len(true_boxes),
        }
        for variant_name, selected_views in VARIANTS.items():
            fused = fuse_predictions(
                predictions_by_view,
                selected_views=selected_views,
                fusion_iou=args.fusion_iou,
            )
            append_stats(
                stats_by_variant[variant_name],
                fused,
                true_boxes,
                true_masks,
            )
            row[f"{variant_name}_detections"] = len(fused)
        count_rows.append(row)

        if image_index % 20 == 0 or image_index == len(image_paths):
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"Processed: {image_index}/{len(image_paths)} "
                f"| elapsed={elapsed:.1f} min"
            )

    records = {}
    for variant_name in VARIANTS:
        records[variant_name] = calculate_metrics(
            stats_by_variant[variant_name],
            output_dir / variant_name,
            make_plots=not smoke_test,
        )

    baseline = records["original"]
    add_comparison_fields(baseline, None)
    for variant_name in VARIANTS:
        if variant_name != "original":
            add_comparison_fields(records[variant_name], baseline)

    builtin_baseline = None
    metric_parity_passed = None
    metric_parity_differences = None
    if not smoke_test:
        print("Running authoritative Ultralytics baseline parity check...")
        builtin_baseline = builtin_baseline_validation(
            model,
            model_path,
            device,
            args.batch,
            args.imgsz,
            output_dir,
        )
        metric_parity_passed, metric_parity_differences = parity_check(
            baseline,
            builtin_baseline,
        )

        for variant_name, record in records.items():
            record["approved"] = (
                False
                if variant_name == "original"
                else approval_decision(record, metric_parity_passed)
            )
    else:
        for record in records.values():
            record["approved"] = False

    eligible = [
        name
        for name, record in records.items()
        if record["approved"]
    ]
    selected_variant = (
        max(
            eligible,
            key=lambda name: records[name]["mean_map50"],
        )
        if eligible
        else "original"
    )

    fieldnames = list(count_rows[0])
    with (output_dir / "image_prediction_counts.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(count_rows)

    save_comparison_csv(output_dir / "comparison.csv", records)

    summary = {
        "model": str(model_path),
        "validation_only": True,
        "test_split_accessed": False,
        "smoke_test": smoke_test,
        "validation_images": len(image_paths),
        "validation_objects": int(
            sum(row["ground_truth_objects"] for row in count_rows)
        ),
        "imgsz": args.imgsz,
        "fusion_iou": args.fusion_iou,
        "views": {
            "0": "original",
            "1": "horizontal_flip_restored_to_original_coordinates",
            "2": "mild_clahe_contrast_no_geometry_change",
        },
        "fusion_confidence": (
            "sum of matched confidences divided by enabled view count"
        ),
        "approval_rules": {
            "minimum_mean_map50_gain": MIN_MEAN_MAP50_GAIN,
            "maximum_mean_map50_95_drop": MAX_MEAN_MAP50_95_DROP,
            "maximum_mean_recall_drop": MAX_RECALL_DROP,
            "maximum_combined_f2_drop": MAX_F2_DROP,
            "minimum_box_precision_at_f2_point": MIN_BOX_PRECISION,
            "minimum_mask_precision_at_f2_point": MIN_MASK_PRECISION,
        },
        "builtin_baseline": builtin_baseline,
        "metric_parity_passed": metric_parity_passed,
        "metric_parity_differences": metric_parity_differences,
        "variants": records,
        "selected_variant": selected_variant,
        "tta_approved_for_qt": bool(eligible),
        "decision": (
            "SMOKE_TEST_ONLY"
            if smoke_test
            else (
                f"APPROVE_{selected_variant}"
                if eligible
                else "KEEP_ORIGINAL_SINGLE_INFERENCE"
            )
        ),
        "elapsed_minutes": (time.time() - start_time) / 60.0,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tta_selection.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "selected_variant": selected_variant,
                "approved_for_qt": bool(eligible),
                "decision": summary["decision"],
                "test_split_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTTA evaluation complete:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults directory: {output_dir}")
    if smoke_test:
        print("Smoke test only: no TTA deployment decision was made.")
    elif eligible:
        print(f"Approved for Qt: {selected_variant}")
    else:
        print("TTA was not approved. Keep original single inference.")


if __name__ == "__main__":
    main()
