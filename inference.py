import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from runtime_metrics import append_jsonl, build_inference_metrics
from model.resnet18 import ResNet18


DEFAULT_WEIGHTS_PATH = "/app/artifacts/final_model.pt"
DEFAULT_MAPPING_PATH = "/app/data/mapping_path.json"
DEFAULT_CLASS_LABELS_PATH = "/app/data/tiny-imagenet-200/words.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Get the first supported backend for quantized PyTorch inference
def get_quantized_engine():
    supported = torch.backends.quantized.supported_engines
    for engine in ["qnnpack", "fbgemm", "onednn"]:
        if engine in supported:
            return engine
    raise RuntimeError(f"No supported quantized backend found. Supported: {supported}")


# Load a regular model from saved weights
def initialize_model(num_classes, weights_path, device):
    if weights_path is None:
        raise ValueError("No weights provided. Please provide model weights for inference.")
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


# Load the compressed model for CPU quantized inference
def load_quantized_model(num_classes, quantized_model_path):
    if not os.path.isfile(quantized_model_path):
        raise FileNotFoundError(f"Model weights file not found: {quantized_model_path}")

    quantized_engine = get_quantized_engine()
    torch.backends.quantized.engine = quantized_engine

    model = ResNet18(num_classes=num_classes)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(quantized_engine)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(
        torch.load(quantized_model_path, weights_only=False, map_location="cpu")
    )
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


# Choose regular or quantized model loading
def load_model(num_classes, weights_path, device, quantized):
    if quantized:
        return load_quantized_model(num_classes, weights_path)

    return initialize_model(
        num_classes=num_classes,
        weights_path=weights_path,
        device=device,
    )


# Run batch inference and store class index plus confidence
@torch.inference_mode()
def predict_batches(model, batches, device):
    predictions = []

    for batch_tensor, batch_paths in batches:
        batch_tensor = batch_tensor.to(device)
        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)

        for image_path, predicted_class, confidence in zip(
            batch_paths,
            predicted_classes.cpu().tolist(),
            confidences.cpu().tolist(),
        ):
            predictions.append(
                {
                    "image_path": image_path,
                    "predicted_class_idx": int(predicted_class),
                    "confidence": float(confidence),
                }
            )

    return predictions


# Convert raw model output into a printable prediction
def format_prediction(predicted_class_idx, confidence, train_id_to_class_id, class_id_to_label):
    class_id = train_id_to_class_id.get(predicted_class_idx)
    if class_id is None:
        raise KeyError(f"No class ID mapping found for train index {predicted_class_idx}.")

    class_label = class_id_to_label.get(class_id)
    if class_label is None:
        raise KeyError(f"No class label mapping found for class ID {class_id}.")

    return {
        "predicted_class_idx": predicted_class_idx,
        "class_id": class_id,
        "predicted_label": class_label,
        "confidence": confidence,
    }


# CLI arguments for batch inference
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run batched inference on image input.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image or a directory of images to classify.",
    )
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--mapping-path", default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--class-labels-path", default=DEFAULT_CLASS_LABELS_PATH)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metrics-log-path",
        default="",
        help="Optional JSONL file for deployment runtime inference metrics.",
    )
    parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load weights as a quantized compressed model.",
    )
    return parser.parse_args(argv)


# Inference CLI entrypoint
def main():
    args = parse_args()
    image_paths = collect_image_paths(args.input)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"
    if args.quantized:
        device = "cpu"

    model = load_model(
        num_classes=args.num_classes,
        weights_path=args.weights_path,
        device=device,
        quantized=args.quantized,
    )

    batches = create_batches(
        image_paths=image_paths,
        batch_size=args.batch_size,
        transform=build_transform(),
    )
    predictions = predict_batches(
        model=model,
        batches=batches,
        device=device,
    )
    if args.metrics_log_path:
        metrics = build_inference_metrics(
            predictions=predictions,
            batch_size=args.batch_size,
            device=device,
            quantized=args.quantized,
            low_confidence_threshold=args.low_confidence_threshold,
            signal_threshold=args.signal_threshold,
        )
        metrics_path = append_jsonl(metrics, args.metrics_log_path)
        print(f"Runtime metrics logged to: {metrics_path}")

    train_id_to_class_id = build_train_id_to_class_id_map(args.mapping_path)
    class_id_to_label = build_class_id_to_label_map(args.class_labels_path)

    print(
        f"Running inference on {len(image_paths)} image(s) "
        f"with batch_size={args.batch_size}, quantized={args.quantized}"
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


if __name__ == "__main__":
    main()
