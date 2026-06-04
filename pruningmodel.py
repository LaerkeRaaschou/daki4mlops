import torch
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from torchvision import transforms

from model.resnet18 import ResNet18
from data.dataloader import get_dataset, get_loaders, get_test_loader


def initialize_model(num_classes, weights_path):
    model = ResNet18(num_classes)
    model = torch.compile(model, backend="eager")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model = model._orig_mod
    return model


def prune_model(model, amount):
    # Collect all conv and linear weights, then prune the lowest-magnitude
    # weights globally across the whole model
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def remove_pruning(model):
    # Make the pruning permanent (bake the mask into the weights)
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, "weight")
    return model


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, train_ids in test_loader:
            images = images.to(device)
            train_ids = train_ids.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == train_ids).sum().item()
            total += train_ids.size(0)

    return correct / total


def finetune_model(model, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}. Train Loss = {avg_loss:.4f}. Test Accuracy = {acc:.2%}")

    return model


def main():
    num_classes = 200
    weights_path = "model/trained_models/resnet_18_classifier_best_acc_epocha44.pt"
    batch_size = 128
    data_path = "data/tiny-imagenet-200/val/images"
    annotations = "data/tiny-imagenet-200/val/val_annotations.txt"
    train_dir = "data/tiny-imagenet-200/train"
    mapping_path = "data/mapping_path.json"
    save_plot_path = "pruning_accuracy.png"
    save_model_path = "model/pruned_models/resnet_18_pruned_finetuned.pt"

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # D4.3: prune to increasing sparsity levels and record accuracy
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracies = []

    for amount in sparsity_levels:
        # Reload a fresh model each level so pruning is applied to the full
        # trained weights and does not compound across levels
        model = initialize_model(num_classes, weights_path)
        if amount > 0.0:
            model = prune_model(model, amount)
        model.to(device)

        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        print(f"Sparsity: {amount:.0%}, Accuracy: {acc:.2%}")

    plt.figure()
    plt.plot(
        [s * 100 for s in sparsity_levels],
        [a * 100 for a in accuracies],
        marker="o",
    )
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Pruning degree vs accuracy")
    plt.grid(True)
    plt.savefig(save_plot_path)
    print(f"\nPruning plot saved to '{save_plot_path}'")

    # D4.4: fine-tune a strongly pruned model to recover the lost accuracy
    prune_amount = 0.8
    finetune_epochs = 5
    finetune_lr = 0.001

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = get_dataset(
        train_dir=train_dir, transform=transform_train, mapping_path=None
    )
    train_loader, _, _ = get_loaders(
        train_set=train_dataset,
        val_set=train_dataset,
        batch_size=batch_size,
        val_split_size=0.1,
        seed=seed,
    )

    # Prune, then evaluate before fine-tuning. The mask stays attached during
    # fine-tuning so the pruned weights remain zero (sparsity is kept)
    model = initialize_model(num_classes, weights_path)
    model = prune_model(model, prune_amount)
    model.to(device)

    acc_before = evaluate(model, test_loader, device)
    print(
        f"\nPruned to {prune_amount:.0%}. Accuracy before fine-tuning = {acc_before:.2%}"
    )

    model = finetune_model(
        model, train_loader, test_loader, finetune_epochs, finetune_lr, device
    )

    # Bake the mask in only after fine-tuning is done
    model = remove_pruning(model)

    acc_after = evaluate(model, test_loader, device)
    print(f"Pruned to {prune_amount:.0%}. Accuracy after fine-tuning = {acc_after:.2%}")
    print(f"Recovered accuracy = {acc_after - acc_before:.2%}")

    torch.save(model.state_dict(), save_model_path)
    print(f"Fine-tuned pruned model saved to '{save_model_path}'")


if __name__ == "__main__":
    main()
