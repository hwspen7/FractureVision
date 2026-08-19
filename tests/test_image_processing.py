from pathlib import Path

import cv2
import numpy as np

import fracture_inference as inference


def test_serializable_detection_removes_nonserializable_visual_fields():
    detection = {
        "confidence": 0.87,
        "box": [10, 20, 30, 40],
        "label": "fracture",
        "mask": np.ones((5, 5), dtype=bool),
        "color_bgr": (0, 255, 0),
    }

    result = inference.serializable_detection(detection)

    assert result["confidence"] == 0.87
    assert result["box"] == [10, 20, 30, 40]
    assert result["label"] == "fracture"

    assert "mask" not in result
    assert "color_bgr" not in result


def test_serializable_detection_does_not_mutate_input():
    detection = {
        "confidence": 0.5,
        "mask": np.ones((2, 2), dtype=bool),
        "color_bgr": (1, 2, 3),
    }

    inference.serializable_detection(detection)

    assert "mask" in detection
    assert "color_bgr" in detection


def test_save_png_creates_valid_png(tmp_path):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[8:24, 12:36] = 255

    output_path = tmp_path / "test.png"

    inference.save_png(output_path, image)

    assert output_path.is_file()

    loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert loaded is not None
    assert loaded.shape == image.shape


def test_resize_radiograph_for_display_preserves_aspect_ratio():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = inference.resize_radiograph_for_display(
        image,
        target_height=400,
    )

    assert result.shape[0] == 400
    assert result.shape[1] == 800


def test_resize_radiograph_for_display_handles_downscaling():
    image = np.zeros((1200, 600, 3), dtype=np.uint8)

    result = inference.resize_radiograph_for_display(
        image,
        target_height=300,
    )

    assert result.shape[:2] == (300, 150)


def test_resize_radiograph_for_display_handles_single_pixel_width():
    image = np.zeros((100, 1, 3), dtype=np.uint8)

    result = inference.resize_radiograph_for_display(
        image,
        target_height=10,
    )

    assert result.shape[0] == 10
    assert result.shape[1] >= 1
