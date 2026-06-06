import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from carbontracker.tracker import CarbonTracker

from publish_monitoring_metrics import post_json
from runtime_metrics import append_jsonl, build_inference_metrics
from model.resnet18 import ResNet18


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Load a regular model from saved weights
def initialize_model(num_classes, weights_path, device):
    if weights_path is None:
        raise ValueError(
            "No weights provided. Please provide model weights for inference."
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    model = ResNet18(num_classes)
    weights = torch.load(weights_path, map_location=device)

    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in weights):
        weights = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in weights.items()
        }

    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model


# Image transform used before inference
def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# Collect one image or all supported images in a folder
def collect_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input image or directory not found: {input_path}")

    image_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No image files found in directory: {input_path}")
    return image_paths


# Open image files and convert them to RGB tensors
def preprocess_image(image_path, transform):
    try:
        with Image.open(image_path) as image:
            return transform(image.convert("RGB")).unsqueeze(0)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Input image not found: {image_path}") from error
    except UnidentifiedImageError as error:
        raise ValueError(f"Input file is not a readable image: {image_path}") from error


# Group images into batches for inference
def create_batches(image_paths, batch_size, transform):
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}.")

    batch_tensors = []
    batch_paths = []

    for image_path in image_paths:
        batch_tensors.append(preprocess_image(image_path, transform).squeeze(0))
        batch_paths.append(image_path)

        if len(batch_tensors) == batch_size:
            yield torch.stack(batch_tensors), batch_paths
            batch_tensors = []
            batch_paths = []

    if batch_tensors:
        yield torch.stack(batch_tensors), batch_paths


# Build train index to Tiny ImageNet class id mapping
def build_train_id_to_class_id_map(mapping_file):
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Class index mapping file not found: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as file:
        class_id_to_train_id = json.load(file)

    train_id_to_class_id = {
        int(train_id): class_id for class_id, train_id in class_id_to_train_id.items()
    }
    if not train_id_to_class_id:
        raise ValueError(f"Class index mapping file is empty: {mapping_file}")
    return train_id_to_class_id


# Build Tiny ImageNet class id to readable label mapping
def build_class_id_to_label_map(mapping_file):
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Class label mapping file not found: {mapping_file}")

    class_id_to_label = {}
    with open(mapping_file, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("\t", maxsplit=1)
            if len(fields) >= 2:
                class_id_to_label[fields[0]] = fields[1]

    if not class_id_to_label:
        raise ValueError(f"Class label mapping file is empty: {mapping_file}")
    return class_id_to_label


# Run batch inference and store class index plus confidence
@torch.inference_mode()
def predict_batches(model, batches, device, on_batch_predictions=None):
    predictions = []

    for batch_tensor, batch_paths in batches:
        batch_tensor = batch_tensor.to(device)
        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        batch_predictions = []

        for image_path, predicted_class, confidence in zip(
            batch_paths,
            predicted_classes.cpu().tolist(),
            confidences.cpu().tolist(),
        ):
            prediction = {
                "image_path": image_path,
                "predicted_class_idx": int(predicted_class),
                "confidence": float(confidence),
            }
            predictions.append(prediction)
            batch_predictions.append(prediction)

        if on_batch_predictions:
            on_batch_predictions(batch_predictions)

    return predictions


# Convert raw model output into a printable prediction
def format_prediction(
    predicted_class_idx, confidence, train_id_to_class_id, class_id_to_label
):
    class_id = train_id_to_class_id.get(predicted_class_idx)
    if class_id is None:
        raise KeyError(
            f"No class ID mapping found for train index {predicted_class_idx}."
        )

    class_label = class_id_to_label.get(class_id)
    if class_label is None:
        raise KeyError(f"No class label mapping found for class ID {class_id}.")

    return {
        "predicted_class_idx": predicted_class_idx,
        "class_id": class_id,
        "predicted_label": class_label,
        "confidence": confidence,
    }


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    image_paths = collect_image_paths(cfg.inference.data_path)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    model = initialize_model(
        num_classes=cfg.inference.num_classes,
        weights_path=cfg.inference.weights_path,
        device=device,
    )

    batches = create_batches(
        image_paths=image_paths,
        batch_size=cfg.inference.batch_size,
        transform=build_transform(),
    )

    tracker = None
    if cfg.carbontracker:
        tracker = CarbonTracker(epochs=1, components="gpu")

    monitoring_url = str(
        OmegaConf.select(cfg, "inference.monitoring_url", default="")
    ).rstrip("/")

    def handle_batch_predictions(batch_predictions):
        metrics = build_inference_metrics(
            predictions=batch_predictions,
            batch_size=cfg.inference.batch_size,
            device=device,
            quantized=False,
            low_confidence_threshold=cfg.inference.low_confidence_threshold,
            signal_threshold=cfg.inference.signal_threshold,
        )
        if cfg.inference.metrics_log_path:
            metrics_path = append_jsonl(metrics, cfg.inference.metrics_log_path)
            print(f"Runtime metrics logged to: {metrics_path}")
        post_json(f"{monitoring_url}/runtime-metrics", metrics)

    batch_callback = handle_batch_predictions if monitoring_url else None

    if cfg.carbontracker:
        tracker.epoch_start()

    predictions = predict_batches(
        model=model,
        batches=batches,
        device=device,
        on_batch_predictions=batch_callback,
    )

    if cfg.carbontracker:
        tracker.epoch_end()

    if cfg.inference.metrics_log_path and not monitoring_url:
        metrics = build_inference_metrics(
            predictions=predictions,
            batch_size=cfg.inference.batch_size,
            device=device,
            quantized=False,
            low_confidence_threshold=cfg.inference.low_confidence_threshold,
            signal_threshold=cfg.inference.signal_threshold,
        )
        metrics_path = append_jsonl(metrics, cfg.inference.metrics_log_path)
        print(f"Runtime metrics logged to: {metrics_path}")

    train_id_to_class_id = build_train_id_to_class_id_map(cfg.inference.mapping_path)
    class_id_to_label = build_class_id_to_label_map(cfg.inference.class_labels_path)

    print(
        f"Running inference on {len(image_paths)} image(s) "
        f"with batch_size={cfg.inference.batch_size}"
    )
    for raw_prediction in predictions:
        prediction = format_prediction(
            predicted_class_idx=raw_prediction["predicted_class_idx"],
            confidence=raw_prediction["confidence"],
            train_id_to_class_id=train_id_to_class_id,
            class_id_to_label=class_id_to_label,
        )
        print(
            f"{Path(raw_prediction['image_path']).name} -> "
            f"{prediction['predicted_label']} "
            f"(class_id={prediction['class_id']}, "
            f"class_idx={prediction['predicted_class_idx']}, "
            f"confidence={prediction['confidence']:.4f})"
        )

    if tracker:
        tracker.stop()


if __name__ == "__main__":
    main()
