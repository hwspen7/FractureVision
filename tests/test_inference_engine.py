from types import SimpleNamespace

import numpy as np
import pytest

import fracture_inference as inference


class FakeYOLOModel:
    def __init__(self, task="segment", results=None):
        self.task = task
        self.results = results if results is not None else []
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return self.results


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fracturevision-test-model")
    return path


def install_fake_yolo(monkeypatch, model, loaded_paths=None):
    def factory(path):
        if loaded_paths is not None:
            loaded_paths.append(path)
        return model

    monkeypatch.setattr(inference, "YOLO", factory)


def test_engine_rejects_missing_model(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(
        FileNotFoundError,
        match="Model not found",
    ):
        inference.FractureInferenceEngine(
            model_path=missing,
            device="cpu",
        )


def test_engine_rejects_non_segmentation_model(
    monkeypatch,
    model_file,
):
    model = FakeYOLOModel(task="detect")
    install_fake_yolo(monkeypatch, model)

    with pytest.raises(
        RuntimeError,
        match="Expected segmentation model",
    ):
        inference.FractureInferenceEngine(
            model_path=model_file,
            device="cpu",
        )


def test_engine_initializes_model_and_configuration(
    monkeypatch,
    model_file,
):
    model = FakeYOLOModel(task="segment")
    loaded_paths = []

    install_fake_yolo(
        monkeypatch,
        model,
        loaded_paths,
    )

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
        imgsz=768,
        fusion_iou=0.55,
        technical_confidence_floor=0.002,
        maximum_detections=50,
    )

    assert engine.model is model
    assert engine.device == "cpu"
    assert engine.imgsz == 768
    assert engine.fusion_iou == pytest.approx(0.55)
    assert engine.technical_confidence_floor == pytest.approx(
        0.002
    )
    assert engine.maximum_detections == 50

    assert loaded_paths == [
        str(model_file.resolve())
    ]


def test_predict_rejects_invalid_numpy_shape(
    monkeypatch,
    model_file,
):
    model = FakeYOLOModel(task="segment")
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
    )

    grayscale = np.zeros(
        (20, 30),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="HxWx3 BGR image",
    ):
        engine.predict(grayscale)

    assert model.predict_calls == []


def test_predict_rejects_unsupported_source_type(
    monkeypatch,
    model_file,
):
    model = FakeYOLOModel(task="segment")
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
    )

    with pytest.raises(
        TypeError,
        match="source must be",
    ):
        engine.predict(123)

    assert model.predict_calls == []


def test_predict_requires_exactly_three_tta_results(
    monkeypatch,
    model_file,
):
    model = FakeYOLOModel(
        task="segment",
        results=[
            SimpleNamespace(),
            SimpleNamespace(),
        ],
    )
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
    )

    image = np.zeros(
        (20, 30, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        RuntimeError,
        match="Expected 3 TTA results",
    ):
        engine.predict(image)

    assert len(model.predict_calls) == 1


def test_predict_runs_complete_tta_pipeline(
    monkeypatch,
    model_file,
):
    result_tokens = [
        object(),
        object(),
        object(),
    ]

    model = FakeYOLOModel(
        task="segment",
        results=result_tokens,
    )
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
        imgsz=704,
        fusion_iou=0.50,
        technical_confidence_floor=0.003,
        maximum_detections=75,
    )

    image = np.zeros(
        (20, 30, 3),
        dtype=np.uint8,
    )
    image[:, :10] = 25

    contrast_image = np.full_like(
        image,
        77,
    )

    monkeypatch.setattr(
        inference,
        "mild_contrast",
        lambda source: contrast_image.copy(),
    )

    conversion_calls = []

    def fake_result_to_predictions(
        result,
        view_id,
        original_height,
        original_width,
    ):
        conversion_calls.append(
            (
                result,
                view_id,
                original_height,
                original_width,
            )
        )

        return [
            {
                "view_id": view_id,
            }
        ]

    monkeypatch.setattr(
        inference,
        "result_to_predictions",
        fake_result_to_predictions,
    )

    mask = np.zeros(
        (20, 30),
        dtype=bool,
    )
    mask[4:12, 6:15] = True

    primary_candidate = {
        "box": np.asarray(
            [4, 2, 17, 14],
            dtype=np.float32,
        ),
        "source_average_box": np.asarray(
            [5, 3, 16, 13],
            dtype=np.float32,
        ),
        "confidence": 0.60,
        "mask": mask,
        "support": 3,
        "absorbed_duplicates": 1,
    }

    suppressed_candidate = {
        "box": np.asarray(
            [5, 3, 16, 13],
            dtype=np.float32,
        ),
        "source_average_box": np.asarray(
            [5, 3, 16, 13],
            dtype=np.float32,
        ),
        "confidence": 0.10,
        "mask": mask.copy(),
        "support": 1,
    }

    fused_candidates = [
        primary_candidate,
        suppressed_candidate,
    ]

    def fake_fuse(
        predictions_by_view,
        fusion_iou,
    ):
        assert set(predictions_by_view) == {
            0,
            1,
            2,
        }

        for view_id in (0, 1, 2):
            assert predictions_by_view[
                view_id
            ][0]["view_id"] == view_id

        assert fusion_iou == pytest.approx(0.50)

        return fused_candidates

    monkeypatch.setattr(
        inference,
        "fuse_predictions",
        fake_fuse,
    )

    def fake_remove_visual_duplicates(predictions):
        assert predictions is fused_candidates

        return (
            [primary_candidate],
            [suppressed_candidate],
        )

    monkeypatch.setattr(
        inference,
        "remove_visual_duplicates",
        fake_remove_visual_duplicates,
    )

    annotation_calls = []

    def fake_annotate(source, detections):
        annotation_calls.append(
            (
                source.copy(),
                detections,
            )
        )

        output = source.copy()
        output[:] = len(detections)
        return output

    monkeypatch.setattr(
        inference,
        "annotate_image",
        fake_annotate,
    )

    output = engine.predict(image)

    assert len(model.predict_calls) == 1

    predict_call = model.predict_calls[0]

    views = predict_call["source"]

    assert len(views) == 3

    np.testing.assert_array_equal(
        views[0],
        image,
    )
    np.testing.assert_array_equal(
        views[1],
        image[:, ::-1],
    )
    np.testing.assert_array_equal(
        views[2],
        contrast_image,
    )

    assert predict_call["imgsz"] == 704
    assert predict_call["batch"] == 3
    assert predict_call["device"] == "cpu"
    assert predict_call["conf"] == pytest.approx(
        0.003
    )
    assert predict_call["iou"] == pytest.approx(
        0.70
    )
    assert predict_call["max_det"] == 75
    assert predict_call["retina_masks"] is False
    assert predict_call["save"] is False
    assert predict_call["verbose"] is False

    assert len(conversion_calls) == 3

    for view_id, call in enumerate(
        conversion_calls
    ):
        result, actual_view_id, height, width = call

        assert result is result_tokens[view_id]
        assert actual_view_id == view_id
        assert height == 20
        assert width == 30

    assert output["detection_count"] == 1
    assert output["raw_fused_candidate_count"] == 2
    assert output["suppressed_duplicate_count"] == 1
    assert output["device"] == "cpu"
    assert output["model_path"] == str(
        model_file.resolve()
    )

    assert output["method"] == (
        "single_model_original_flip_mild_contrast_wbf_"
        "visual_deduplication"
    )

    assert output[
        "evidence_normalization"
    ]["is_medical_probability"] is False

    detections = output["detections"]

    assert len(detections) == 1

    detection = detections[0]

    expected_evidence = (
        inference.normalized_evidence_score(
            0.60
        )
    )

    assert detection["number"] == 1
    assert detection["confidence"] == pytest.approx(
        0.60
    )
    assert detection[
        "raw_fusion_confidence"
    ] == pytest.approx(0.60)
    assert detection[
        "confidence_percent"
    ] == pytest.approx(60.0)
    assert detection[
        "evidence_score"
    ] == pytest.approx(expected_evidence)
    assert detection["display_label"] == (
        f"#1 {expected_evidence * 100:.2f}%"
    )
    assert detection["box"] == [
        4.0,
        2.0,
        17.0,
        14.0,
    ]
    assert detection["model_fused_box"] == [
        5.0,
        3.0,
        16.0,
        13.0,
    ]
    assert detection["view_support"] == 3
    assert detection["absorbed_duplicates"] == 1
    assert detection["mask"] is mask

    assert len(annotation_calls) == 1
    assert len(annotation_calls[0][1]) == 1

    assert len(output["annotated_images"]) == 1
    assert (
        output["annotated_image"]
        is output["annotated_images"][0]
    )

    np.testing.assert_array_equal(
        output["original_image"],
        image,
    )


def test_predict_empty_result_creates_single_blank_page(
    monkeypatch,
    model_file,
):
    empty_result = SimpleNamespace(
        boxes=None,
        masks=None,
    )

    model = FakeYOLOModel(
        task="segment",
        results=[
            empty_result,
            empty_result,
            empty_result,
        ],
    )
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
    )

    annotation_calls = []

    def fake_annotate(image, detections):
        annotation_calls.append(detections)
        return image.copy()

    monkeypatch.setattr(
        inference,
        "annotate_image",
        fake_annotate,
    )

    image = np.zeros(
        (20, 30, 3),
        dtype=np.uint8,
    )

    output = engine.predict(image)

    assert output["detections"] == []
    assert output["detection_count"] == 0
    assert output["raw_fused_candidate_count"] == 0
    assert output["suppressed_duplicate_count"] == 0

    assert annotation_calls == [[]]

    assert len(
        output["annotated_images"]
    ) == 1

    assert (
        output["annotated_image"]
        is output["annotated_images"][0]
    )


def test_predict_file_path_uses_read_image(
    monkeypatch,
    model_file,
):
    empty_result = SimpleNamespace(
        boxes=None,
        masks=None,
    )

    model = FakeYOLOModel(
        task="segment",
        results=[
            empty_result,
            empty_result,
            empty_result,
        ],
    )
    install_fake_yolo(monkeypatch, model)

    engine = inference.FractureInferenceEngine(
        model_path=model_file,
        device="cpu",
    )

    image = np.zeros(
        (16, 24, 3),
        dtype=np.uint8,
    )

    requested_paths = []

    def fake_read_image(path):
        requested_paths.append(path)
        return image.copy()

    monkeypatch.setattr(
        inference,
        "read_image",
        fake_read_image,
    )

    monkeypatch.setattr(
        inference,
        "annotate_image",
        lambda source, detections: source.copy(),
    )

    output = engine.predict("example_xray.png")

    assert requested_paths == [
        "example_xray.png"
    ]

    assert output["original_image"].shape == (
        16,
        24,
        3,
    )

    assert len(model.predict_calls) == 1
