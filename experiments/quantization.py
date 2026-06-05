import datetime
import os
import sys
from pathlib import Path

import torch
import torch.nn
import torch.quantization
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import get_test_loader
from model.resnet18 import ResNet18
from test import report_statistics, test_model


def get_quantized_engine():
    for engine in ["qnnpack", "fbgemm", "onednn", "x86"]:
        if engine in torch.backends.quantized.supported_engines:
            return engine
    raise RuntimeError(
        f"No supported quantized backend found: "
        f"{torch.backends.quantized.supported_engines}"
    )


def quantize_model(full_model, quantized_model):
    torch.backends.quantized.engine = get_quantized_engine()
    model = ResNet18(num_classes=200)
    model = torch.compile(model, backend="eager")
    model.load_state_dict(torch.load(full_model, map_location=torch.device("cpu")))
    model = model._orig_mod
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    model_prepared = torch.quantization.prepare(model, inplace=False)
    with torch.no_grad():
        for _ in range(10):
            input_tensor = torch.randn(1, 3, 64, 64, dtype=torch.float)
            model_prepared(input_tensor)
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    torch.save(model_quantized.state_dict(), quantized_model)
    print("Quantized model success")


def load_quantized_model(quantized_model_path):
    torch.backends.quantized.engine = get_quantized_engine()
    model = ResNet18(num_classes=200)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(
        torch.load(quantized_model_path, weights_only=False, map_location="cpu")
    )
    return model


def run_inference(model_path, data, q):
    if q:
        model = load_quantized_model(model_path)
    else:
        model = ResNet18(num_classes=200)
        model = torch.compile(model, backend="eager")
        model.load_state_dict(
            torch.load(model_path, weights_only=False, map_location="cpu")
        )
        model = model._orig_mod
    model.eval()
    with torch.no_grad():
        _ = model(data)  # warmup
    start = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            _ = model(data)
    end = datetime.datetime.now()
    return (end - start).total_seconds()


def test(model_path, version):
    if version:
        model = load_quantized_model(model_path)
    else:
        model = ResNet18(num_classes=200)
        model = torch.compile(model, backend="eager")
        model.load_state_dict(
            torch.load(model_path, weights_only=False, map_location="cpu")
        )
        model = model._orig_mod
    num_classes = 200
    batch_size = 1
    data_path = "data/tiny-imagenet-200/val/images"
    annotations = "data/tiny-imagenet-200/val/val_annotations.txt"
    save_stats_path = f"test_statistics_{version}.txt"
    mapping_path = "data/mapping_path.json"
    test_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    test_loader = get_test_loader(
        mapping_path=mapping_path,
        test_dir=data_path,
        transform_test=test_transform,
        test_annotations=annotations,
        batch_size=batch_size,
        shuffle=False,
    )
    start = datetime.datetime.now()
    total_stats, class_stats = test_model(model, num_classes, test_loader)
    end = datetime.datetime.now()
    report_statistics(total_stats, class_stats, save_stats_path)
    return (end - start).total_seconds()


def main():
    full_model = "model/trained_models/resnet_18_classifier_best_acc_epocha44.pt"
    quantized_model = "model/quantized_models/quantized44.pt"
    device = "cpu"
    data = torch.randn(1, 3, 64, 64, dtype=torch.float).to(device)
    if os.path.exists(quantized_model):
        f_time = run_inference(full_model, data, q=False)
        q_time = run_inference(quantized_model, data, q=True)
        print("Difference between models in seconds (dummy data)")
        print("float32:", f_time, "  qint8:", q_time)
        f_time = test(model_path=full_model, version=False)
        q_time = test(model_path=quantized_model, version=True)
        print("Difference between models in seconds (test data)")
        print("float32:", f_time, "  qint8:", q_time)
    else:
        quantize_model(full_model, quantized_model)
        f_time = run_inference(full_model, data, q=False)
        q_time = run_inference(quantized_model, data, q=True)
        print("float32:", f_time, "  qint8:", q_time)


if __name__ == "__main__":
    main()
