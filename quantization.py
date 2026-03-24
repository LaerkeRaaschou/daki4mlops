import torch
import torch.quantization
import torch.nn
import os
import datetime
from model.resnet18 import ResNet18

torch.backends.quantized.engine = "qnnpack"


def quantize_model(full_model, quantized_model):
    torch.backends.quantized.engine = "qnnpack"
    model = ResNet18(num_classes=200)
    model = torch.compile(model, backend="eager")
    model.load_state_dict(torch.load(full_model, map_location=torch.device("cpu")))
    model = model._orig_mod
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model_prepared = torch.quantization.prepare(model, inplace=False)
    with torch.no_grad():
        for _ in range(10):
            input_tensor = torch.randn(1, 3, 64, 64, dtype=torch.float)
            model_prepared(input_tensor)
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    torch.save(model_quantized.state_dict(), quantized_model)
    print("Quantized model success")


def load_quantized_model(quantized_model_path):
    torch.backends.quantized.engine = "qnnpack"
    model = ResNet18(num_classes=200)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
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
        _ = model(data)
    end = datetime.datetime.now()
    return (end - start).microseconds


def main():
    full_model = "model/trained_models/resnet_18_classifier_best_acc_epocha44.pt"
    quantized_model = "model/quantized_models/quantized44.pt"
    device = "cpu"
    data = torch.randn(1, 3, 64, 64, dtype=torch.float).to(device)
    if os.path.exists(quantized_model):
        f_time = run_inference(full_model, data, q=False)
        q_time = run_inference(quantized_model, data, q=True)
        print("Difference between models in microseconds")
        print("float32:", f_time, "  qint8:", q_time)
    else:
        quantize_model(full_model, quantized_model)
        f_time = run_inference(full_model, data, q=False)
        q_time = run_inference(quantized_model, data, q=True)
        print("float32:", f_time, "  qint8:", q_time)


if __name__ == "__main__":
    main()
