import os
import csv
import json

import hydra
import torch
from hydra.utils import to_absolute_path
from torchvision import transforms
from PIL import Image

from model.resnet18 import ResNet18


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


def load_inputs(dir_path):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Input image directory not found: {dir_path}")

    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def preprocess_sample(sample_path, transform):
    image = Image.open(sample_path).convert("RGB")
    tensor = transform(image)
    metadata = {
        "path": sample_path,
        "filename": os.path.basename(sample_path),
    }

    return tensor, metadata


def create_batches(samples, batch_size, transform):
    batch_tensors = []
    batch_metadata = []

    for sample in samples:
        tensor, metadata = preprocess_sample(sample, transform)

        batch_tensors.append(tensor)
        batch_metadata.append(metadata)

        if len(batch_tensors) == batch_size:
            yield torch.stack(batch_tensors), batch_metadata
            batch_tensors = []
            batch_metadata = []

    if batch_tensors:
        yield torch.stack(batch_tensors), batch_metadata


@torch.inference_mode()
def inference(model, batches, device):
    predictions = []

    for batch_tensor, batch_metadata in batches:
        batch_tensor = batch_tensor.to(device)

        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)

        for metadata, pred_class, confidence in zip(
            batch_metadata, predicted_classes, confidences
        ):
            predictions.append(
                {
                    "path": metadata["path"],
                    "filename": metadata["filename"],
                    "predicted_class_idx": pred_class.item(),
                    "confidence": confidence.item(),
                }
            )

    return predictions


def build_train_id_to_class_id_map(mapping_file):
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Class index mapping file not found: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as file:
        class_id_to_train_id = json.load(file)

    if not isinstance(class_id_to_train_id, dict):
        raise ValueError(
            "Class index mapping file must be JSON in the form "
            "{class_id: train_id}."
        )

    train_id_to_class_id = {
        int(train_id): class_id for class_id, train_id in class_id_to_train_id.items()
    }

    if not train_id_to_class_id:
        raise ValueError(f"Class index mapping file is empty: {mapping_file}")

    return train_id_to_class_id


def build_class_id_to_label_map(mapping_file):
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Class label mapping file not found: {mapping_file}")

    class_id_to_label = {}

    with open(mapping_file, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("\t", maxsplit=1)
            if len(fields) >= 2:
                class_id = fields[0]
                class_label = fields[1]
                class_id_to_label[class_id] = class_label

    if not class_id_to_label:
        raise ValueError(f"Class label mapping file is empty: {mapping_file}")

    return class_id_to_label


def postprocess_predictions(predictions, train_id_to_class_id, class_id_to_label):
    final_predictions = []

    for pred in predictions:
        train_id = pred["predicted_class_idx"]
        class_id = train_id_to_class_id.get(train_id)
        if class_id is None:
            raise KeyError(f"No class ID mapping found for train index {train_id}.")

        class_label = class_id_to_label.get(class_id)
        if class_label is None:
            raise KeyError(f"No class label mapping found for class ID {class_id}.")

        final_predictions.append(
            {
                "path": pred["path"],
                "filename": pred["filename"],
                "predicted_class_idx": train_id,
                "class_id": class_id,
                "predicted_label": class_label,
                "confidence": round(pred["confidence"], 4),
            }
        )

    return final_predictions


def output_predictions(predictions, output_path="predictions.csv"):
    if not predictions:
        print("No predictions to output.")
        return

    print(f"Total predictions: {len(predictions)}")
    print("Example predictions:")

    for pred in predictions[:5]:
        print(
            f"{pred['filename']} -> "
            f"{pred['predicted_label']} "
            f"(class_idx={pred['predicted_class_idx']}, "
            f"confidence={pred['confidence']})"
        )

    fieldnames = [
        "path",
        "filename",
        "predicted_class_idx",
        "class_id",
        "predicted_label",
        "confidence",
    ]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Predictions saved to: {output_path}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    data_path = to_absolute_path(str(cfg.inference.input_path))
    weights_path = to_absolute_path(str(cfg.inference.weights_path))
    batch_size = cfg.data.batch_size
    num_classes = cfg.inference.num_classes
    train_id_to_class_id_path = to_absolute_path(str(cfg.data.mapping_path))
    class_id_to_label_path = to_absolute_path(str(cfg.inference.class_labels_path))
    output_path = to_absolute_path(str(cfg.inference.output_path))

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    samples = load_inputs(data_path)
    if not samples:
        print(f"No image files found in: {data_path}")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    batches = create_batches(samples, batch_size, transform)

    model = initialize_model(
        num_classes=num_classes,
        weights_path=weights_path,
        device=device,
    )

    predictions = inference(model, batches, device)

    train_id_to_class_id = build_train_id_to_class_id_map(train_id_to_class_id_path)
    class_id_to_label = build_class_id_to_label_map(class_id_to_label_path)

    final = postprocess_predictions(
        predictions,
        train_id_to_class_id,
        class_id_to_label,
    )

    output_predictions(final, output_path)


if __name__ == "__main__":
    main()
