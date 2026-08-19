from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fracture_inference as inference


class FakeBoxes:
    def __init__(self, boxes, confidences, classes):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confidences, dtype=torch.float32)
        self.cls = torch.tensor(classes, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def make_result(
    boxes,
    confidences,
    classes,
    masks,
):
    fake_boxes = FakeBoxes(
        boxes,
        confidences,
        classes,
    )

    fake_masks = SimpleNamespace(
        data=torch.from_numpy(
            np.asarray(masks, dtype=np.float32)
        )
    )

    return SimpleNamespace(
        boxes=fake_boxes,
        masks=fake_masks,
    )


def make_prediction(
    box,
    confidence,
    mask,
    view_id,
):
    return {
        "box": np.asarray(box, dtype=np.float32),
        "confidence": float(confidence),
        "mask": np.asarray(mask, dtype=bool),
        "view_id": view_id,
    }


def make_fused_prediction(
    box,
    confidence,
    mask,
):
    return {
        "box": np.asarray(box, dtype=np.float32),
        "source_average_box": np.asarray(
            box,
            dtype=np.float32,
        ),
        "confidence": float(confidence),
        "mask": np.asarray(mask, dtype=bool),
        "support": 1,
    }


def test_result_to_predictions_returns_empty_when_no_boxes():
    result = SimpleNamespace(
        boxes=None,
        masks=None,
    )

    predictions = inference.result_to_predictions(
        result,
        view_id=0,
        original_height=20,
        original_width=30,
    )

    assert predictions == []


def test_result_to_predictions_requires_segmentation_masks():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[2, 3, 8, 9]],
            confidences=[0.8],
            classes=[0],
        ),
        masks=None,
    )

    with pytest.raises(
        RuntimeError,
        match="boxes without masks",
    ):
        inference.result_to_predictions(
            result,
            view_id=0,
            original_height=20,
            original_width=30,
        )


def test_result_to_predictions_rejects_box_mask_count_mismatch():
    result = make_result(
        boxes=[
            [2, 3, 8, 9],
            [10, 10, 15, 15],
        ],
        confidences=[0.8, 0.7],
        classes=[0, 0],
        masks=[
            np.zeros((20, 30), dtype=np.float32),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="box/mask count mismatch",
    ):
        inference.result_to_predictions(
            result,
            view_id=0,
            original_height=20,
            original_width=30,
        )


def test_result_to_predictions_rejects_unexpected_class():
    mask = np.zeros((20, 30), dtype=np.float32)
    mask[3:9, 2:8] = 1.0

    result = make_result(
        boxes=[[2, 3, 8, 9]],
        confidences=[0.8],
        classes=[1],
        masks=[mask],
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected predicted class",
    ):
        inference.result_to_predictions(
            result,
            view_id=0,
            original_height=20,
            original_width=30,
        )


def test_result_to_predictions_preserves_original_view():
    mask = np.zeros((20, 30), dtype=np.float32)
    mask[3:9, 2:8] = 1.0

    result = make_result(
        boxes=[[2, 3, 8, 9]],
        confidences=[0.8],
        classes=[0],
        masks=[mask],
    )

    predictions = inference.result_to_predictions(
        result,
        view_id=0,
        original_height=20,
        original_width=30,
    )

    assert len(predictions) == 1

    prediction = predictions[0]

    np.testing.assert_allclose(
        prediction["box"],
        [2, 3, 8, 9],
    )

    assert prediction["confidence"] == pytest.approx(
        0.8,
        abs=1e-6,
    )
    assert prediction["view_id"] == 0
    assert prediction["mask"].shape == (20, 30)


def test_result_to_predictions_restores_horizontal_flip():
    mask = np.zeros((10, 20), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    result = make_result(
        boxes=[[2, 2, 6, 6]],
        confidences=[0.75],
        classes=[0],
        masks=[mask],
    )

    predictions = inference.result_to_predictions(
        result,
        view_id=1,
        original_height=10,
        original_width=20,
    )

    prediction = predictions[0]

    np.testing.assert_allclose(
        prediction["box"],
        [14, 2, 18, 6],
    )

    expected_mask = np.ascontiguousarray(
        mask[:, ::-1]
    )

    np.testing.assert_allclose(
        prediction["mask"],
        expected_mask,
    )


def test_fuse_predictions_combines_three_views():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True

    predictions_by_view = {
        0: [
            make_prediction(
                [9, 9, 21, 21],
                0.9,
                mask,
                0,
            )
        ],
        1: [
            make_prediction(
                [9, 9, 21, 21],
                0.6,
                mask,
                1,
            )
        ],
        2: [
            make_prediction(
                [9, 9, 21, 21],
                0.3,
                mask,
                2,
            )
        ],
    }

    fused = inference.fuse_predictions(
        predictions_by_view,
        fusion_iou=0.45,
    )

    assert len(fused) == 1

    prediction = fused[0]

    assert prediction["support"] == 3
    assert prediction["confidence"] == pytest.approx(
        0.6,
        abs=1e-6,
    )

    np.testing.assert_allclose(
        prediction["source_average_box"],
        [9, 9, 21, 21],
        rtol=0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        prediction["box"],
        [8, 8, 22, 22],
    )

    assert prediction["mask"].dtype == bool
    assert prediction["mask"].sum() == 100


def test_fuse_predictions_does_not_merge_same_view_predictions():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True

    predictions_by_view = {
        0: [
            make_prediction(
                [5, 5, 15, 15],
                0.9,
                mask,
                0,
            ),
            make_prediction(
                [5, 5, 15, 15],
                0.8,
                mask,
                0,
            ),
        ],
        1: [],
        2: [],
    }

    fused = inference.fuse_predictions(
        predictions_by_view,
    )

    assert len(fused) == 2
    assert all(
        prediction["support"] == 1
        for prediction in fused
    )


def test_fuse_predictions_keeps_nonoverlapping_regions_separate():
    mask_a = np.zeros((40, 40), dtype=bool)
    mask_a[2:10, 2:10] = True

    mask_b = np.zeros((40, 40), dtype=bool)
    mask_b[25:35, 25:35] = True

    predictions_by_view = {
        0: [
            make_prediction(
                [2, 2, 10, 10],
                0.9,
                mask_a,
                0,
            )
        ],
        1: [
            make_prediction(
                [25, 25, 35, 35],
                0.8,
                mask_b,
                1,
            )
        ],
        2: [],
    }

    fused = inference.fuse_predictions(
        predictions_by_view,
    )

    assert len(fused) == 2


def test_fuse_single_view_confidence_is_divided_by_three():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:10, 4:10] = True

    predictions_by_view = {
        0: [
            make_prediction(
                [4, 4, 10, 10],
                0.9,
                mask,
                0,
            )
        ],
        1: [],
        2: [],
    }

    fused = inference.fuse_predictions(
        predictions_by_view,
    )

    assert len(fused) == 1
    assert fused[0]["support"] == 1
    assert fused[0]["confidence"] == pytest.approx(
        0.3,
        abs=1e-6,
    )


def test_intersection_over_smaller_nested_boxes_is_one():
    outer = np.asarray(
        [0, 0, 20, 20],
        dtype=np.float32,
    )
    inner = np.asarray(
        [5, 5, 10, 10],
        dtype=np.float32,
    )

    result = inference.intersection_over_smaller(
        outer,
        inner,
    )

    assert result == pytest.approx(1.0)


def test_intersection_over_smaller_nonoverlapping_is_zero():
    box_a = np.asarray(
        [0, 0, 10, 10],
        dtype=np.float32,
    )
    box_b = np.asarray(
        [20, 20, 30, 30],
        dtype=np.float32,
    )

    assert inference.intersection_over_smaller(
        box_a,
        box_b,
    ) == pytest.approx(0.0)


def test_intersection_over_smaller_handles_zero_area():
    box_a = np.asarray(
        [5, 5, 5, 10],
        dtype=np.float32,
    )
    box_b = np.asarray(
        [0, 0, 10, 10],
        dtype=np.float32,
    )

    assert inference.intersection_over_smaller(
        box_a,
        box_b,
    ) == pytest.approx(0.0)


def test_normalized_center_distance_identical_boxes_is_zero():
    box = np.asarray(
        [0, 0, 10, 10],
        dtype=np.float32,
    )

    assert inference.normalized_center_distance(
        box,
        box,
    ) == pytest.approx(0.0)


def test_normalized_center_distance_uses_box_diagonal():
    box_a = np.asarray(
        [0, 0, 10, 10],
        dtype=np.float32,
    )
    box_b = np.asarray(
        [10, 0, 20, 10],
        dtype=np.float32,
    )

    expected = 10.0 / np.hypot(10.0, 10.0)

    assert inference.normalized_center_distance(
        box_a,
        box_b,
    ) == pytest.approx(expected)


def test_remove_visual_duplicates_merges_near_identical_predictions():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True

    strong = make_fused_prediction(
        [4, 4, 16, 16],
        0.8,
        mask,
    )
    weak = make_fused_prediction(
        [4, 4, 16, 16],
        0.3,
        mask,
    )

    kept, suppressed = inference.remove_visual_duplicates(
        [strong, weak]
    )

    assert len(kept) == 1
    assert len(suppressed) == 1

    assert kept[0]["confidence"] == pytest.approx(0.8)
    assert kept[0]["absorbed_duplicates"] == 1
    assert kept[0]["absorbed_duplicate_confidences"] == [
        pytest.approx(0.3)
    ]


def test_remove_visual_duplicates_merges_much_weaker_same_region():
    empty_mask = np.zeros((30, 30), dtype=bool)

    strong = make_fused_prediction(
        [0, 0, 10, 10],
        0.8,
        empty_mask,
    )
    weak = make_fused_prediction(
        [1, 1, 9, 9],
        0.1,
        empty_mask,
    )

    kept, suppressed = inference.remove_visual_duplicates(
        [strong, weak]
    )

    assert len(kept) == 1
    assert len(suppressed) == 1
    assert kept[0]["absorbed_duplicates"] == 1


def test_remove_visual_duplicates_preserves_distinct_regions():
    mask_a = np.zeros((40, 40), dtype=bool)
    mask_a[4:10, 4:10] = True

    mask_b = np.zeros((40, 40), dtype=bool)
    mask_b[25:31, 25:31] = True

    first = make_fused_prediction(
        [3, 3, 11, 11],
        0.8,
        mask_a,
    )
    second = make_fused_prediction(
        [24, 24, 32, 32],
        0.7,
        mask_b,
    )

    kept, suppressed = inference.remove_visual_duplicates(
        [first, second]
    )

    assert len(kept) == 2
    assert suppressed == []


def test_remove_visual_duplicates_preserves_original_input_dict():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:10, 4:10] = True

    prediction = make_fused_prediction(
        [3, 3, 11, 11],
        0.8,
        mask,
    )

    kept, _ = inference.remove_visual_duplicates(
        [prediction]
    )

    assert "absorbed_duplicates" not in prediction
    assert "absorbed_duplicate_confidences" not in prediction

    assert kept[0]["absorbed_duplicates"] == 0
    assert kept[0]["absorbed_duplicate_confidences"] == []
