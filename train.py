import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from model.resnet18 import ResNet18
from data.dataloader import get_train_val_loader
import wandb
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score


class EarlyStopping:
    def __init__(self, patience, delta, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(
                    f"No improvement ({self.no_improvement_count}/{self.patience}). "
                    f"Best: {self.best_loss:.4f}, Current: {val_loss:.4f}"
                )
            if self.no_improvement_count > self.patience:
                self.stop_training = True
            return self.stop_training


def train_model(model, criterion, dataloader, optimizer, device, epoch):
    model.train()

    # Initialize
    total_loss = 0.0

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        if epoch == 1 and i == 0:
            print("\nFirst batch:")
            print("Predicted shape:", y_pred.shape)
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 200 == 0:
            print(
                "Train Epoch: %s, Iteration: %s, Train Loss: %s"
                % (epoch, i, loss.item())
            )
            wandb.log({"Train Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    wandb.log({"Train Avg Loss": avg_loss})

    return avg_loss


@torch.no_grad()
def val_model(model, criterion, batches, device, epoch, num_classes=200):
    model.eval()
    labels = list(range(num_classes))
    all_preds = []
    all_targets = []

    total_loss = 0.0

    for x, y in batches:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)

        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / len(batches)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    wandb.log(
        {
            "Val Avg Loss": avg_loss,
            "Val Accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "Epoch": epoch,
        }
    )
    return avg_loss, accuracy, precision, recall


def main():

    # Set variables
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-2
    criterion = nn.CrossEntropyLoss()

    # Data path
    data_path = "data/tiny-imagenet-200"

    # Use device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        print("mps")
    else:
        device = torch.device("cpu")
        print("cpu")

    # Start wandb
    wandb.login()
    wandb.init(project="tiny-imagenet-resnet18")

    # Initialize model
    model = ResNet18(num_classes=200).to(device)

    # Define optimizer
    optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    transform_train = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # Train loader
    train_loader, val_dataset = get_train_val_loader(
        mapping_path="data/mapping_path.json",
        train_dir=f"{data_path}/train",
        batch_size=batch_size,
        transform_train=transform_train,
        val_split_size=10,
    )

    # Data check
    x0, y0 = next(iter(train_loader))
    print("\nData check:")
    print("Input shape:", x0.shape)
    print("Labels shape:", y0.shape)
    print("Label range:", y0.min().item(), "to", y0.max().item())

    # Model check
    out = model(x0.to(device))
    print("\nModel check:")
    print("Output shape:", out.shape)

    early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(
            model, criterion, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_precision, val_recall = val_model(
            model, criterion, val_dataset, device, epoch
        )

        scheduler.step()
        print("Current LR:", optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}. "
            f"Train Loss = {train_loss:.4f}. "
            f"Val Loss = {val_loss:.4f}. "
            f"Val Accuracy = {val_acc:.3f}. "
            f"Val Precision = {val_precision:.3f}."
            f"Val Recall = {val_recall:.3f}."
        )

        early_stopping.check_early_stop(val_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            torch.save(
                model.state_dict(), f"resnet_18_classifier_final_epoch{epoch}.pt"
            )
            break

    torch.save(model.state_dict(), f"resnet_18_classifier_final_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
