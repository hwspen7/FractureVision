import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "segment"
    / "fracatlas_yolo11s_seg_recall_v2"
    / "weights"
    / "f2_best.pt"
)


# Validation-only F2 operating point selected for the approved three-view TTA
# pipeline. It is used only to normalize the UI evidence scale; model outputs
# and evaluation metrics remain unchanged.
TTA_F2_OPERATING_THRESHOLD = 0.13713713713713713


CONFIDENCE_STYLES = (
    {
        "minimum": 0.75,
        "level": "high",
        "level_label": "High",
        "color_bgr": (94, 63, 244),
        "color_hex": "#F43F5E",
    },
    {
        "minimum": 0.50,
        "level": "medium",
        "level_label": "Medium",
        "color_bgr": (11, 158, 245),
        "color_hex": "#F59E0B",
    },
    {
        "minimum": 0.25,
        "level": "low",
        "level_label": "Low",
        "color_bgr": (212, 182, 6),
        "color_hex": "#06B6D4",
    },
    {
        "minimum": 0.00,
        "level": "trace",
        "level_label": "Trace",
        "color_bgr": (148, 163, 184),
        "color_hex": "#94A3B8",
    },
)


def select_device(requested=None):
    if requested:
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def read_image(path):
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as source:
        rgb = np.asarray(
            ImageOps.exif_transpose(source).convert("RGB")
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mild_contrast(image):
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


def confidence_style(confidence):
    for style in CONFIDENCE_STYLES:
        if confidence >= style["minimum"]:
            return style
    return CONFIDENCE_STYLES[-1]


def normalized_evidence_score(
    raw_confidence,
    operating_threshold=TTA_F2_OPERATING_THRESHOLD,
):
    """Map the validated operating threshold to 50% on an evidence scale."""
    epsilon = 1e-6
    raw = float(np.clip(raw_confidence, epsilon, 1.0 - epsilon))
    threshold = float(np.clip(operating_threshold, epsilon, 1.0 - epsilon))
    raw_odds = raw / (1.0 - raw)
    threshold_odds = threshold / (1.0 - threshold)
    normalized_odds = raw_odds / threshold_odds
    return float(normalized_odds / (1.0 + normalized_odds))


def box_iou(box_a, box_b):
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


def weighted_box(members):
    weights = np.asarray(
        [member["confidence"] for member in members],
        dtype=np.float32,
    )
    boxes = np.stack(
        [member["box"] for member in members],
        axis=0,
    )
    return np.average(boxes, axis=0, weights=weights).astype(np.float32)


def mask_bounding_box(mask, fallback_box, padding=2):
    """Return a box that fully contains the final binary mask."""
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return np.asarray(fallback_box, dtype=np.float32).copy()

    image_height, image_width = mask.shape
    x1 = max(0, int(columns.min()) - padding)
    y1 = max(0, int(rows.min()) - padding)
    x2 = min(image_width - 1, int(columns.max()) + padding + 1)
    y2 = min(image_height - 1, int(rows.max()) + padding + 1)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def scale_mask_to_original(mask, original_height, original_width):
    mask_height, mask_width = mask.shape
    gain = min(
        mask_width / original_width,
        mask_height / original_height,
    )
    resized_width = round(original_width * gain)
    resized_height = round(original_height * gain)
    pad_x = (mask_width - resized_width) / 2.0
    pad_y = (mask_height - resized_height) / 2.0

    left = max(0, int(round(pad_x - 0.1)))
    top = max(0, int(round(pad_y - 0.1)))
    right = min(mask_width, int(round(mask_width - pad_x + 0.1)))
    bottom = min(mask_height, int(round(mask_height - pad_y + 0.1)))

    cropped = mask[top:bottom, left:right]
    if cropped.size == 0:
        return np.zeros(
            (original_height, original_width),
            dtype=np.float32,
        )
    return cv2.resize(
        cropped.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def result_to_predictions(
    result,
    view_id,
    original_height,
    original_width,
):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or result.masks.data is None:
        raise RuntimeError(
            "The model produced boxes without masks. "
            "Please load a YOLO segmentation checkpoint."
        )

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    masks = result.masks.data.detach().cpu().numpy().astype(np.float32)

    if len(boxes) != len(masks):
        raise RuntimeError("Prediction box/mask count mismatch.")

    predictions = []
    for box, confidence, class_id, mask in zip(
        boxes,
        confidences,
        classes,
        masks,
    ):
        if int(class_id) != 0:
            raise RuntimeError(f"Unexpected predicted class: {class_id}")

        restored_box = box.copy()
        restored_mask = scale_mask_to_original(
            mask,
            original_height,
            original_width,
        )
        if view_id == 1:
            x1 = float(restored_box[0])
            x2 = float(restored_box[2])
            restored_box[0] = original_width - x2
            restored_box[2] = original_width - x1
            restored_mask = np.ascontiguousarray(restored_mask[:, ::-1])

        restored_box[[0, 2]] = np.clip(
            restored_box[[0, 2]],
            0,
            original_width - 1,
        )
        restored_box[[1, 3]] = np.clip(
            restored_box[[1, 3]],
            0,
            original_height - 1,
        )
        predictions.append(
            {
                "box": restored_box,
                "confidence": float(confidence),
                "mask": restored_mask,
                "view_id": view_id,
            }
        )
    return predictions


def fuse_predictions(predictions_by_view, fusion_iou=0.45):
    all_predictions = []
    for view_id in (0, 1, 2):
        all_predictions.extend(predictions_by_view[view_id])
    all_predictions.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )

    clusters = []
    for prediction in all_predictions:
        best_cluster = None
        best_overlap = fusion_iou
        for cluster in clusters:
            used_views = {
                member["view_id"]
                for member in cluster["members"]
            }
            if prediction["view_id"] in used_views:
                continue
            overlap = box_iou(prediction["box"], cluster["box"])
            if overlap >= best_overlap:
                best_overlap = overlap
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
            best_cluster["box"] = weighted_box(best_cluster["members"])

    fused = []
    for cluster in clusters:
        members = cluster["members"]
        weights = np.asarray(
            [member["confidence"] for member in members],
            dtype=np.float32,
        )
        masks = np.stack(
            [member["mask"] for member in members],
            axis=0,
        )
        fused_confidence = float(weights.sum() / 3.0)
        fused_mask = np.average(
            masks,
            axis=0,
            weights=weights,
        )
        binary_mask = fused_mask >= 0.50
        average_box = weighted_box(members)
        display_box = mask_bounding_box(
            binary_mask,
            fallback_box=average_box,
            padding=2,
        )
        fused.append(
            {
                "box": display_box,
                "source_average_box": average_box,
                "confidence": min(fused_confidence, 1.0),
                "mask": binary_mask,
                "support": len(members),
            }
        )

    fused.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )
    return fused


def intersection_over_smaller(box_a, box_b):
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
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def normalized_center_distance(box_a, box_b):
    center_a = np.asarray(
        [
            (float(box_a[0]) + float(box_a[2])) / 2.0,
            (float(box_a[1]) + float(box_a[3])) / 2.0,
        ],
        dtype=np.float32,
    )
    center_b = np.asarray(
        [
            (float(box_b[0]) + float(box_b[2])) / 2.0,
            (float(box_b[1]) + float(box_b[3])) / 2.0,
        ],
        dtype=np.float32,
    )
    diagonal_a = float(
        np.hypot(
            float(box_a[2]) - float(box_a[0]),
            float(box_a[3]) - float(box_a[1]),
        )
    )
    diagonal_b = float(
        np.hypot(
            float(box_b[2]) - float(box_b[0]),
            float(box_b[3]) - float(box_b[1]),
        )
    )
    reference = max(diagonal_a, diagonal_b, 1.0)
    return float(np.linalg.norm(center_a - center_b) / reference)


def remove_visual_duplicates(predictions):
    """
    Remove only clearly redundant display boxes.

    Near-identical masks and boxes are always merged. A weaker candidate is
    also merged when it lies in the same local region. Mask overlap is used
    whenever possible so that separate nearby fractures are preserved.
    """
    kept = []
    suppressed = []
    for prediction in predictions:
        duplicate_of = None
        for stronger in kept:
            overlap = box_iou(prediction["box"], stronger["box"])
            coverage = intersection_over_smaller(
                prediction["box"],
                stronger["box"],
            )
            center_distance = normalized_center_distance(
                prediction["box"],
                stronger["box"],
            )

            mask_a = np.asarray(prediction["mask"], dtype=bool)
            mask_b = np.asarray(stronger["mask"], dtype=bool)
            mask_area_a = int(mask_a.sum())
            mask_area_b = int(mask_b.sum())
            masks_valid = mask_area_a > 0 and mask_area_b > 0
            if masks_valid:
                mask_intersection = int(np.logical_and(mask_a, mask_b).sum())
                mask_coverage = mask_intersection / min(
                    mask_area_a,
                    mask_area_b,
                )
            else:
                mask_coverage = 0.0

            near_identical = (
                (
                    overlap >= 0.65
                    and center_distance <= 0.30
                    and (not masks_valid or mask_coverage >= 0.35)
                )
                or (masks_valid and mask_coverage >= 0.75)
            )
            much_weaker = prediction["confidence"] <= max(
                0.02,
                stronger["confidence"] * 0.40,
            )
            same_region = (
                (overlap >= 0.45 and center_distance <= 0.40)
                or (coverage >= 0.80 and center_distance <= 0.40)
                or (
                    masks_valid
                    and mask_coverage >= 0.50
                    and center_distance <= 0.45
                )
            )
            if near_identical or (much_weaker and same_region):
                duplicate_of = stronger
                break

        if duplicate_of is None:
            prediction = dict(prediction)
            prediction["absorbed_duplicates"] = 0
            prediction["absorbed_duplicate_confidences"] = []
            kept.append(prediction)
        else:
            duplicate_of["absorbed_duplicates"] += 1
            duplicate_of["absorbed_duplicate_confidences"].append(
                float(prediction["confidence"])
            )
            suppressed.append(prediction)

    return kept, suppressed


def annotate_image(
    image,
    detections,
    mask_alpha=0.30,
    radiograph_height=1200,
):
    source_height, source_width = image.shape[:2]
    scale = radiograph_height / source_height
    image_width = max(1, int(round(source_width * scale)))
    image_height = int(radiograph_height)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    annotated = cv2.resize(
        image,
        (image_width, image_height),
        interpolation=interpolation,
    )
    line_width = 2

    def render_box(detection):
        return np.asarray(detection["box"], dtype=np.float32) * scale

    def render_mask(detection):
        return cv2.resize(
            detection["mask"].astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Draw translucent masks first. Text is deliberately kept outside the
    # radiograph so that it never hides bone structure.
    for detection in reversed(detections):
        mask = render_mask(detection)
        color = np.asarray(detection["color_bgr"], dtype=np.float32)
        if mask.any():
            original_pixels = annotated[mask].astype(np.float32)
            blended = (
                original_pixels * (1.0 - mask_alpha)
                + color * mask_alpha
            )
            annotated[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = np.rint(render_box(detection)).astype(int)
        color = tuple(int(value) for value in detection["color_bgr"])
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            line_width,
            cv2.LINE_AA,
        )

    if not detections:
        return annotated

    # Build a dedicated callout panel to the right of the X-ray. The canvas
    # may grow vertically only when an unusually large number of candidates
    # must be shown; no candidate is silently removed.
    font_scale = 0.52
    font_thickness = 1
    card_height = 38
    card_gap = max(18, round(card_height * 0.55))
    margin = 18
    # Keep the callout area around 15-18% of the X-ray width.
    panel_width = max(150, min(185, round(image_width * 0.18)))
    required_height = (
        margin * 2
        + len(detections) * card_height
        + max(0, len(detections) - 1) * card_gap
    )
    canvas_height = max(image_height, required_height)
    canvas_width = image_width + panel_width
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        (42, 23, 15),
        dtype=np.uint8,
    )
    canvas[:image_height, :image_width] = annotated
    cv2.line(
        canvas,
        (image_width, 0),
        (image_width, canvas_height - 1),
        (75, 85, 99),
        1,
        cv2.LINE_AA,
    )

    # Keep the side panel in confidence order (#1, #2, ...), with a fixed,
    # clearly visible gap between cards.
    ordered = list(detections)
    group_height = (
        len(ordered) * card_height
        + max(0, len(ordered) - 1) * card_gap
    )
    first_top = max(margin, (canvas_height - group_height) // 2)
    desired_tops = [
        first_top + index * (card_height + card_gap)
        for index in range(len(ordered))
    ]

    card_left = image_width + 14
    card_right = canvas_width - 8
    first_elbow_x = image_width + 7

    for route_index, (detection, card_top) in enumerate(zip(
        ordered,
        desired_tops,
    )):
        color = tuple(int(value) for value in detection["color_bgr"])
        x1, y1, x2, y2 = render_box(detection)
        anchor_x = int(round(min(max(x2, 0), image_width - 1)))
        anchor_y = int(
            round(min(max((y1 + y2) / 2.0, 0), image_height - 1))
        )
        card_bottom = card_top + card_height
        card_center_y = int(round((card_top + card_bottom) / 2))
        elbow_x = first_elbow_x + route_index * 6

        cv2.circle(
            canvas,
            (anchor_x, anchor_y),
            max(2, line_width + 1),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (anchor_x, anchor_y),
            (elbow_x, anchor_y),
            color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (elbow_x, anchor_y),
            (elbow_x, card_center_y),
            color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (elbow_x, card_center_y),
            (card_left, card_center_y),
            color,
            line_width,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            canvas,
            (card_left, card_top),
            (card_right, card_bottom),
            color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (card_left, card_top),
            (card_right, card_bottom),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        number_text = f"#{detection['number']}"
        confidence_text = f"{detection['evidence_score_percent']:.2f}%"
        (_, number_height), number_baseline = cv2.getTextSize(
            number_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        (confidence_width, confidence_height), confidence_baseline = (
            cv2.getTextSize(
                confidence_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
        )
        number_x = card_left + 11
        confidence_x = card_right - confidence_width - 11
        number_y = card_top + (
            card_height + number_height - number_baseline
        ) // 2
        confidence_y = card_top + (
            card_height + confidence_height - confidence_baseline
        ) // 2
        cv2.putText(
            canvas,
            number_text,
            (int(number_x), int(number_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            confidence_text,
            (int(confidence_x), int(confidence_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return canvas


class FractureInferenceEngine:
    """Single-checkpoint YOLO11s-Seg inference with validated three-view TTA."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        device=None,
        imgsz=640,
        fusion_iou=0.45,
        technical_confidence_floor=0.001,
        maximum_detections=100,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.device = select_device(device)
        self.imgsz = int(imgsz)
        self.fusion_iou = float(fusion_iou)
        self.technical_confidence_floor = float(technical_confidence_floor)
        self.maximum_detections = int(maximum_detections)

        self.model = YOLO(str(self.model_path))
        if self.model.task != "segment":
            raise RuntimeError(
                f"Expected segmentation model, got task={self.model.task!r}."
            )

    def predict(self, source):
        if isinstance(source, (str, Path)):
            image = read_image(source)
        elif isinstance(source, np.ndarray):
            if source.ndim != 3 or source.shape[2] != 3:
                raise ValueError("NumPy source must be an HxWx3 BGR image.")
            image = np.ascontiguousarray(source.copy())
        else:
            raise TypeError("source must be a file path or an HxWx3 BGR array.")

        image_height, image_width = image.shape[:2]
        views = [
            image,
            np.ascontiguousarray(image[:, ::-1]),
            mild_contrast(image),
        ]
        results = self.model.predict(
            source=views,
            imgsz=self.imgsz,
            batch=3,
            device=self.device,
            conf=self.technical_confidence_floor,
            iou=0.70,
            max_det=self.maximum_detections,
            retina_masks=False,
            save=False,
            verbose=False,
        )
        if len(results) != 3:
            raise RuntimeError(f"Expected 3 TTA results, got {len(results)}.")

        predictions_by_view = {
            view_id: result_to_predictions(
                result,
                view_id,
                image_height,
                image_width,
            )
            for view_id, result in enumerate(results)
        }
        raw_fused = fuse_predictions(
            predictions_by_view,
            fusion_iou=self.fusion_iou,
        )
        fused, suppressed_duplicates = remove_visual_duplicates(raw_fused)

        detections = []
        for number, prediction in enumerate(fused, start=1):
            confidence = float(prediction["confidence"])
            evidence_score = normalized_evidence_score(confidence)
            style = confidence_style(evidence_score)
            box = [
                float(value)
                for value in prediction["box"]
            ]
            model_fused_box = [
                float(value)
                for value in prediction["source_average_box"]
            ]
            detections.append(
                {
                    "number": number,
                    "display_label": f"#{number} {evidence_score * 100:.2f}%",
                    "confidence": confidence,
                    "confidence_percent": confidence * 100.0,
                    "raw_fusion_confidence": confidence,
                    "raw_fusion_confidence_percent": confidence * 100.0,
                    "evidence_score": evidence_score,
                    "evidence_score_percent": evidence_score * 100.0,
                    "confidence_level": style["level"],
                    "confidence_level_label": style["level_label"],
                    "color_bgr": style["color_bgr"],
                    "color_hex": style["color_hex"],
                    "box": box,
                    "box_source": "final_fused_mask_bounds_with_padding",
                    "model_fused_box": model_fused_box,
                    "mask": prediction["mask"],
                    "view_support": int(prediction["support"]),
                    "absorbed_duplicates": int(
                        prediction.get("absorbed_duplicates", 0)
                    ),
                }
            )

        # One independent candidate per page: boxes, masks, leader lines, and
        # labels therefore cannot overlap even when an image has several
        # separate fracture candidates.
        annotated_images = [
            annotate_image(image, [detection])
            for detection in detections
        ]
        if not annotated_images:
            annotated_images = [annotate_image(image, [])]

        return {
            "original_image": image,
            # Backward-compatible first page for simple callers.
            "annotated_image": annotated_images[0],
            "annotated_images": annotated_images,
            "detections": detections,
            "detection_count": len(detections),
            "raw_fused_candidate_count": len(raw_fused),
            "suppressed_duplicate_count": len(suppressed_duplicates),
            "device": str(self.device),
            "model_path": str(self.model_path),
            "method": (
                "single_model_original_flip_mild_contrast_wbf_"
                "visual_deduplication"
            ),
            "evidence_normalization": {
                "name": "validation_threshold_odds_normalization",
                "tta_f2_operating_threshold": TTA_F2_OPERATING_THRESHOLD,
                "operating_threshold_maps_to": 0.50,
                "is_medical_probability": False,
            },
        }


def serializable_detection(detection):
    return {
        key: value
        for key, value in detection.items()
        if key not in {"mask", "color_bgr"}
    }


def save_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode image: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise RuntimeError(f"Could not save image: {path}") from error


def resize_radiograph_for_display(image, target_height=1200):
    source_height, source_width = image.shape[:2]
    scale = target_height / source_height
    target_width = max(1, int(round(source_width * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (target_width, int(target_height)),
        interpolation=interpolation,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run one-image YOLO11s-Seg TTA inference preview."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fracture_analysis_result"),
        help=(
            "Output directory. If a file suffix is supplied, only the suffix "
            "is removed and a directory is still created."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    engine = FractureInferenceEngine(
        model_path=args.model,
        device=args.device,
    )
    prediction = engine.predict(args.input)

    requested_output = args.output.expanduser().resolve()
    output_directory = (
        requested_output.with_suffix("")
        if requested_output.suffix
        else requested_output
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    original_path = output_directory / "original.png"
    display_original = resize_radiograph_for_display(
        prediction["original_image"],
        target_height=1200,
    )
    save_png(original_path, display_original)

    result_paths = []
    if prediction["detections"]:
        for index, annotated_image in enumerate(
            prediction["annotated_images"],
            start=1,
        ):
            result_path = output_directory / f"candidate_{index:02d}.png"
            save_png(result_path, annotated_image)
            result_paths.append(result_path)
    else:
        result_path = output_directory / "no_candidate.png"
        save_png(result_path, prediction["annotated_images"][0])
        result_paths.append(result_path)

    serialized_detections = []
    for index, detection in enumerate(prediction["detections"]):
        record = serializable_detection(detection)
        record["result_image"] = result_paths[index].name
        serialized_detections.append(record)

    report = {
        "model": prediction["model_path"],
        "device": prediction["device"],
        "method": prediction["method"],
        "evidence_normalization": prediction["evidence_normalization"],
        "detection_count": prediction["detection_count"],
        "raw_fused_candidate_count": prediction[
            "raw_fused_candidate_count"
        ],
        "suppressed_duplicate_count": prediction[
            "suppressed_duplicate_count"
        ],
        "source_image": str(args.input.expanduser().resolve()),
        "output_directory": str(output_directory),
        "original_image": original_path.name,
        "detections": serialized_detections,
        "result_images": [path.name for path in result_paths],
        "confidence_note": (
            "Evidence Score is validation-threshold normalized and is not a "
            "medical probability. Raw Fusion is preserved for transparency."
        ),
    }
    report_path = output_directory / "result.json"
    report["result_json"] = report_path.name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
