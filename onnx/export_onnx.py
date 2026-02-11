import torch
from inference.onnx.resnet18 import ResNet18   # <-- your file

NUM_CLASSES = 200
H, W = 64, 64

def main():
    model = ResNet18(num_classes=NUM_CLASSES)

    # state = torch.load("resnet_18_classifier.pt", map_location="cpu")
    # state = {k.replace("module.", ""): v for k, v in state.items()}
    # model.load_state_dict(state, strict=True)

    model.eval()

    dummy = torch.randn(1, 3, H, W)

    torch.onnx.export(
        model,
        dummy,
        "inference\\onnx\\resnet18_tinyimagenet_64.onnx",
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print("ONNX export complete")

def check_onnx():
    import onnx
    import onnxruntime as ort

    path = "inference\\onnx\\resnet18_tinyimagenet_64.onnx"

    m = onnx.load(path)
    onnx.checker.check_model(m)
    print("onnx.checker: OK")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("ORT load: OK")
    print("Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

if __name__ == "__main__":
    main()

    check = True
    if check:
        check_onnx()