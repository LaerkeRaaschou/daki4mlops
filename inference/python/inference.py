import sys
import os
import csv

import hydra
import torch
from torchvision import transforms
from PIL import Image

from model.resnet18 import ResNet18

def initialize_model(num_classes, weights_path, device):
    model = ResNet18(num_classes)
    if weights_path is None:
        print("No weights provided. Please provide model weights for inference.")
        sys.exit(1)
    
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    
    model.eval()    
    model.to(device)

    return model

def load_inputs(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

def preprocess_sample(sample_path, transform):  
    image = Image.open(sample_path).convert('RGB')
    tensor = transform(image)
    metadata = {
        'path': sample_path,
        'filename': os.path.basename(sample_path),
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

        for metadata, pred_class, confidence in zip(batch_metadata, predicted_classes, confidences):
            predictions.append({
                "path": metadata["path"],
                "filename": metadata["filename"],
                "predicted_class_idx": pred_class.item(),
                "confidence": confidence.item(),
            })

    return predictions

def build_train_id_to_class_id_map(mapping_file):
    train_id_to_class_id = {}

    with open(mapping_file, "r") as file:
        for line in file:
            fields = line.strip().split("\t")
            if len(fields) >= 2:
                class_id = fields[0]
                train_id = int(fields[1])
                train_id_to_class_id[train_id] = class_id

    return train_id_to_class_id

def build_class_id_to_label_map(mapping_file):
    class_id_to_label = {}

    with open(mapping_file, "r") as file:
        for line in file:
            fields = line.strip().split("\t")
            if len(fields) >= 2:
                class_id = fields[0]
                class_label = fields[1]
                class_id_to_label[class_id] = class_label

    return class_id_to_label

def postprocess_predictions(predictions, train_id_to_class_id, class_id_to_label):
    final_predictions = []

    for pred in predictions:
        train_id = pred["predicted_class_idx"]
        class_id = train_id_to_class_id.get(train_id)
        class_label = class_id_to_label.get(class_id, str(class_id))

        final_predictions.append({
            "path": pred["path"],
            "filename": pred["filename"],
            "predicted_class_idx": train_id,
            "class_id": class_id,
            "predicted_label": class_label,
            "confidence": round(pred["confidence"], 4),
        })

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

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Predictions saved to: {output_path}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    data_path = cfg.data.root
    weights_path = cfg.model.weights_path
    batch_size = cfg.data.batch_size
    train_id_to_class_id_path = cfg.data.mapping_path
    #class_id_to_label_path = cfg.model.class_id_label_map
    class_id_to_label_path = "data\tiny-imagenet-200\words.txt"

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
        num_classes=200,
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

    output_predictions(final)


if __name__ == "__main__":
    main()