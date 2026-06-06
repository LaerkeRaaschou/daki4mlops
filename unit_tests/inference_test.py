import json

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

import inference


def test_preprocess_image_accepts_readable_image(tmp_path):
    image_path = tmp_path / "sample.JPEG"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)

    tensor = inference.preprocess_image(image_path, inference.build_transform())

    assert tuple(tensor.shape) == (1, 3, 64, 64)


def test_preprocess_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.preprocess_image(tmp_path / "missing.png", inference.build_transform())


def test_preprocess_image_rejects_non_image(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        inference.preprocess_image(text_path, inference.build_transform())


def test_collect_image_paths_accepts_single_image(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)

    assert inference.collect_image_paths(image_path) == [image_path]


def test_collect_image_paths_finds_images_in_folder(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.JPEG"
    ignored = tmp_path / "notes.txt"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(first)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(second)
    ignored.write_text("not an image", encoding="utf-8")

    assert inference.collect_image_paths(tmp_path) == [first, second]


def test_create_batches_yields_final_partial_batch(tmp_path):
    paths = []
    for index in range(3):
        image_path = tmp_path / f"{index}.png"
        Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
        paths.append(image_path)

    batches = list(inference.create_batches(paths, batch_size=2, transform=inference.build_transform()))

    assert len(batches) == 2
    assert tuple(batches[0][0].shape) == (2, 3, 64, 64)
    assert tuple(batches[1][0].shape) == (1, 3, 64, 64)


def test_build_train_id_to_class_id_map(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"n0001": 0, "n0002": 1}), encoding="utf-8")

    assert inference.build_train_id_to_class_id_map(mapping_path) == {
        0: "n0001",
        1: "n0002",
    }


def test_format_prediction_maps_label():
    prediction = inference.format_prediction(
        predicted_class_idx=1,
        confidence=0.75,
        train_id_to_class_id={1: "n0002"},
        class_id_to_label={"n0002": "example label"},
    )

    assert prediction["predicted_label"] == "example label"
    assert prediction["class_id"] == "n0002"
    assert prediction["confidence"] == 0.75


def test_predict_batches_calls_callback_once_per_batch():
    class FakeModel:
        def __call__(self, batch_tensor):
            return torch.tensor(
                [[0.0, 2.0], [3.0, 0.0], [0.5, 1.5]],
                dtype=torch.float32,
            )[: batch_tensor.shape[0]]

    batches = [
        (torch.zeros((2, 3, 64, 64)), ["a.png", "b.png"]),
        (torch.zeros((1, 3, 64, 64)), ["c.png"]),
    ]
    callback_batches = []

    predictions = inference.predict_batches(
        model=FakeModel(),
        batches=batches,
        device="cpu",
        on_batch_predictions=lambda batch_predictions: callback_batches.append(
            list(batch_predictions)
        ),
    )

    assert len(callback_batches) == 2
    assert [prediction["image_path"] for prediction in predictions] == [
        "a.png",
        "b.png",
        "c.png",
    ]
    assert [prediction["image_path"] for prediction in callback_batches[0]] == [
        "a.png",
        "b.png",
    ]
    assert [prediction["image_path"] for prediction in callback_batches[1]] == ["c.png"]


def test_inference_carbontracker_wraps_prediction(monkeypatch):
    calls = []

    class FakeTracker:
        def __init__(self, epochs, components):
            calls.append(("init", epochs, components))

        def epoch_start(self):
            calls.append(("epoch_start",))

        def epoch_end(self):
            calls.append(("epoch_end",))

        def stop(self):
            calls.append(("stop",))

    cfg = OmegaConf.create(
        {
            "carbontracker": True,
            "device": "cpu",
            "inference": {
                "data_path": "images",
                "weights_path": "model.pt",
                "num_classes": 200,
                "batch_size": 32,
                "metrics_log_path": "",
                "mapping_path": "mapping.json",
                "class_labels_path": "words.txt",
                "monitoring_url": "",
                "low_confidence_threshold": 0.5,
                "signal_threshold": 0.5,
            },
        }
    )

    monkeypatch.setattr(inference, "collect_image_paths", lambda path: ["image.png"])
    monkeypatch.setattr(inference, "initialize_model", lambda **kwargs: "model")
    monkeypatch.setattr(inference, "build_transform", lambda: "transform")
    monkeypatch.setattr(
        inference,
        "create_batches",
        lambda image_paths, batch_size, transform: ["batch"],
    )

    def fake_predict_batches(model, batches, device, on_batch_predictions=None):
        calls.append(("predict_batches",))
        return [
            {"image_path": "image.png", "predicted_class_idx": 1, "confidence": 0.75}
        ]

    monkeypatch.setattr(inference, "predict_batches", fake_predict_batches)
    monkeypatch.setattr(
        inference,
        "build_train_id_to_class_id_map",
        lambda mapping_file: {1: "n0002"},
    )
    monkeypatch.setattr(
        inference,
        "build_class_id_to_label_map",
        lambda mapping_file: {"n0002": "example label"},
    )
    monkeypatch.setattr(inference, "CarbonTracker", FakeTracker)

    inference.main.__wrapped__(cfg)

    assert calls == [
        ("init", 1, "gpu"),
        ("epoch_start",),
        ("predict_batches",),
        ("epoch_end",),
        ("stop",),
    ]


def test_inference_posts_runtime_metrics_per_batch_without_final_aggregate(monkeypatch):
    appended_records = []
    posted_records = []

    cfg = OmegaConf.create(
        {
            "carbontracker": False,
            "device": "cpu",
            "inference": {
                "data_path": "images",
                "weights_path": "model.pt",
                "num_classes": 200,
                "batch_size": 32,
                "metrics_log_path": "runtime_metrics.jsonl",
                "mapping_path": "mapping.json",
                "class_labels_path": "words.txt",
                "monitoring_url": "http://monitoring:8000",
                "low_confidence_threshold": 0.5,
                "signal_threshold": 0.5,
            },
        }
    )

    first_batch = [
        {"image_path": "a.png", "predicted_class_idx": 1, "confidence": 0.9}
    ]
    second_batch = [
        {"image_path": "b.png", "predicted_class_idx": 1, "confidence": 0.2}
    ]

    monkeypatch.setattr(inference, "collect_image_paths", lambda path: ["a.png", "b.png"])
    monkeypatch.setattr(inference, "initialize_model", lambda **kwargs: "model")
    monkeypatch.setattr(inference, "build_transform", lambda: "transform")
    monkeypatch.setattr(
        inference,
        "create_batches",
        lambda image_paths, batch_size, transform: ["first", "second"],
    )

    def fake_predict_batches(model, batches, device, on_batch_predictions=None):
        on_batch_predictions(first_batch)
        on_batch_predictions(second_batch)
        return first_batch + second_batch

    def fake_append_jsonl(record, output_path):
        appended_records.append((record, output_path))
        return output_path

    def fake_post_json(url, payload):
        posted_records.append((url, payload))

    monkeypatch.setattr(inference, "predict_batches", fake_predict_batches)
    monkeypatch.setattr(inference, "append_jsonl", fake_append_jsonl)
    monkeypatch.setattr(inference, "post_json", fake_post_json)
    monkeypatch.setattr(
        inference,
        "build_train_id_to_class_id_map",
        lambda mapping_file: {1: "n0002"},
    )
    monkeypatch.setattr(
        inference,
        "build_class_id_to_label_map",
        lambda mapping_file: {"n0002": "example label"},
    )

    inference.main.__wrapped__(cfg)

    assert len(appended_records) == 2
    assert len(posted_records) == 2
    assert [url for url, _payload in posted_records] == [
        "http://monitoring:8000/runtime-metrics",
        "http://monitoring:8000/runtime-metrics",
    ]
    assert [record["image_count"] for record, _path in appended_records] == [1, 1]
    assert [record["batch_count"] for record, _path in appended_records] == [1, 1]
    assert [record["average_confidence"] for record, _path in appended_records] == [
        0.9,
        0.2,
    ]


