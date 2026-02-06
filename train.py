import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from model.resnet18 import ResNet18
from data.dataloader import get_train_loader
import wandb
from torchvision import transforms



def train_model(model, dataloader, optimizer, device, epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss()

    # Initialize
    total_loss = 0.0

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)

        if epoch == 1 and i == 0:
            print("\nFirst batch:")
            print("Predicted shape:", y_pred.shape)
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 200 == 0:
            print("Train Epoch: %s, Iteration: %s, Train Loss: %s" % (epoch, i, loss.item()))
            wandb.log({"Train Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    wandb.log({"Train Avg Loss": avg_loss})

    return avg_loss

@torch.no_grad()
def val_model(model, batches, device, epoch):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        
        y_pred = model(x)
        loss = loss_function(y_pred, y)

        total_loss += loss.item()

        preds = torch.argmax(y_pred, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / len(batches)
    acc = total_correct / total_samples

    wandb.log({"Val Avg Loss": avg_loss, "Val Accuracy": acc, "Epoch": epoch})
    return avg_loss, acc



def main():

    # Set variables
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-2
    

    # Data path
    data_path = "data/tiny-imagenet-200"


    # Use device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")


    # Start wandb
    wandb.login()
    wandb.init(project="tiny-imagenet-resnet18")


    # Initialize model 
    model = ResNet18(num_classes=200).to(device)

    # Define optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    transform_train = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                               std=(0.229, 0.224, 0.225))])
    # Train loader
    train_loader = get_train_loader(train_dir=f"{data_path}/train", batch_size=batch_size, transform_train=transform_train, shuffle=True)

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

    # Validation batch, 2 
    mini_val = []
    for i, (x, y) in enumerate(train_loader):
        mini_val.append((x, y))
        if i == 1:
            break

    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = val_model(model, mini_val, device, epoch)

        scheduler.step()
        print("Current LR:", optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}. "
            f"Train Loss = {train_loss:.4f}. "
            f"Val Loss = {val_loss:.4f}. "
            f"Val Accuracy = {val_acc:.3f}. "
        )

    torch.save(model.state_dict(), "resnet_18_classifier.pt")


if __name__ == "__main__":
    main()
