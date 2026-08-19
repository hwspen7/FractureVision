import numpy as np
import pytest

import fracture_inference as inference


def test_confidence_style_returns_matching_style():
    for confidence in (0.0, 0.25, 0.5, 0.75, 1.0):
        style = inference.confidence_style(confidence)

        expected = next(
            (
                candidate
                for candidate in inference.CONFIDENCE_STYLES
                if confidence >= candidate["minimum"]
            ),
            inference.CONFIDENCE_STYLES[-1],
        )

        assert style == expected


def test_normalized_evidence_score_maps_threshold_to_half():
    threshold = inference.TTA_F2_OPERATING_THRESHOLD

    score = inference.normalized_evidence_score(
        threshold,
        operating_threshold=threshold,
    )

    assert score == pytest.approx(0.5, abs=1e-6)


def test_normalized_evidence_score_is_monotonic():
    threshold = inference.TTA_F2_OPERATING_THRESHOLD

    low = inference.normalized_evidence_score(
        threshold * 0.5,
        operating_threshold=threshold,
    )
    middle = inference.normalized_evidence_score(
        threshold,
        operating_threshold=threshold,
    )
    high = inference.normalized_evidence_score(
        min(0.999, threshold * 1.5),
        operating_threshold=threshold,
    )

    assert low < middle
    assert middle < high


def test_normalized_evidence_score_stays_in_unit_interval():
    for confidence in (-1.0, 0.0, 0.2, 0.8, 1.0, 2.0):
        score = inference.normalized_evidence_score(confidence)
        assert 0.0 <= score <= 1.0


def test_box_iou_identical_boxes():
    box = np.asarray([10, 20, 50, 70], dtype=np.float32)

    assert inference.box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_non_overlapping_boxes():
    box_a = np.asarray([0, 0, 10, 10], dtype=np.float32)
    box_b = np.asarray([20, 20, 30, 30], dtype=np.float32)

    assert inference.box_iou(box_a, box_b) == pytest.approx(0.0)


def test_box_iou_partial_overlap():
    box_a = np.asarray([0, 0, 10, 10], dtype=np.float32)
    box_b = np.asarray([5, 5, 15, 15], dtype=np.float32)

    expected = 25.0 / 175.0

    assert inference.box_iou(box_a, box_b) == pytest.approx(expected)


def test_box_iou_handles_zero_area_box():
    box_a = np.asarray([5, 5, 5, 10], dtype=np.float32)
    box_b = np.asarray([0, 0, 10, 10], dtype=np.float32)

    assert inference.box_iou(box_a, box_b) == pytest.approx(0.0)


def test_weighted_box_uses_confidence_weights():
    members = [
        {
            "box": np.asarray([0, 0, 10, 10], dtype=np.float32),
            "confidence": 1.0,
        },
        {
            "box": np.asarray([10, 10, 20, 20], dtype=np.float32),
            "confidence": 3.0,
        },
    ]

    result = inference.weighted_box(members)

    expected = np.asarray(
        [7.5, 7.5, 17.5, 17.5],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float32


def test_mask_bounding_box_uses_fallback_for_empty_mask():
    mask = np.zeros((20, 30), dtype=np.uint8)
    fallback = np.asarray([2, 3, 10, 12], dtype=np.float32)

    result = inference.mask_bounding_box(mask, fallback)

    np.testing.assert_array_equal(result, fallback)

    assert result.dtype == np.float32
    assert result is not fallback


def test_mask_bounding_box_contains_mask_with_padding():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 8:15] = 1

    result = inference.mask_bounding_box(
        mask,
        fallback_box=[0, 0, 1, 1],
        padding=2,
    )

    expected = np.asarray(
        [6, 3, 17, 12],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(result, expected)


def test_mask_bounding_box_clamps_to_image_boundaries():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:3, 0:3] = 1

    result = inference.mask_bounding_box(
        mask,
        fallback_box=[0, 0, 1, 1],
        padding=5,
    )

    assert result[0] == 0
    assert result[1] == 0
    assert result[2] <= 9
    assert result[3] <= 9


def test_scale_mask_to_original_restores_requested_shape():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0

    result = inference.scale_mask_to_original(
        mask,
        original_height=40,
        original_width=80,
    )

    assert result.shape == (40, 80)
    assert result.dtype == np.float32


def test_scale_mask_to_original_preserves_nonempty_region():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0

    result = inference.scale_mask_to_original(
        mask,
        original_height=32,
        original_width=48,
    )

    assert result.shape == (32, 48)
    assert np.max(result) > 0
