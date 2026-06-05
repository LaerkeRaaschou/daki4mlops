import json
from types import SimpleNamespace

import pytest
from PIL import Image

import inference


def test_preprocess_image_accepts_readable_image(tmp_path):
    image_path = tmp_path / "sample.JPEG"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)

    tensor = inference.preprocess_image(image_path, inference.build_transform())

    assert tuple(tensor.shape) == (1, 3, 64, 64)


def test_preprocess_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.preprocess_image(
            tmp_path / "missing.png", inference.build_transform()
        )


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

    batches = list(
        inference.create_batches(
            paths, batch_size=2, transform=inference.build_transform()
        )
    )

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


def test_parse_args_exposes_optional_runtime_metrics(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference.py",
            "--input",
            "images",
            "--metrics-log-path",
            "runtime/inference.jsonl",
        ],
    )

    args = inference.parse_args()

    assert args.metrics_log_path == "runtime/inference.jsonl"
    assert args.low_confidence_threshold == 0.5
    assert args.signal_threshold == 0.5


def test_parse_args_defaults_to_hydra_inference_config():
    cfg = SimpleNamespace(
        device="cuda",
        inference=SimpleNamespace(
            data_path="/configured/images",
            weights_path="/configured/model.pt",
            mapping_path="/configured/mapping.json",
            class_labels_path="/configured/words.txt",
            num_classes=200,
            batch_size=64,
        ),
    )

    args = inference.parse_args([], cfg=cfg)

    assert args.input == "/configured/images"
    assert args.weights_path == "/configured/model.pt"
    assert args.mapping_path == "/configured/mapping.json"
    assert args.class_labels_path == "/configured/words.txt"
    assert args.num_classes == 200
    assert args.batch_size == 64
    assert args.device == "cuda"
