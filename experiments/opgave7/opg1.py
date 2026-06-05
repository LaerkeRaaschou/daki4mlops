"""
Exercises/Tasks:

Since continual learning and unlearning are still some quite immature fields within AI we will
fallback to use the good old MNIST dataset to learn about and implement solutions to these concepts.
It is okay to use the Pytorch MNIST example located here.

Exercise 1: Continual Learning with Experience Replay

Objective:
Understand and implement continual learning using experience replay to prevent catastrophic forgetting. You will first observe how naive sequential training leads to forgetting before applying replay as a solution.

Instructions:

Task 1 - Initial Training:
Train a neural network on the first five digits of MNIST (0–4).
Evaluate and record the accuracy on this subset.

Task 2 - Naïve Sequential Learning:
Train the same model on the second half of MNIST (5–9) without resetting weights and without using experience replay.
Evaluate accuracy on both old (0–4) and new (5–9) digits.
Print/plot the performance drop on digits 0–4 as the training progresses, showing catastrophic forgetting.
"""

import argparse
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader, label=""):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\n{}Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            f"{label} " if label else "",
            test_loss,
            correct,
            len(test_loader.dataset),
            accuracy,
        )
    )
    return accuracy


def filter_by_digits(dataset, digits):
    """Return a Subset of `dataset` containing only the given digit classes."""
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    mask = torch.isin(targets, torch.tensor(list(digits)))
    indices = torch.where(mask)[0].tolist()
    return Subset(dataset, indices)


def plot_accuracy_history(
    history, title="Catastrophic Forgetting", save_path="forgetting.png"
):
    """history: dict like {"0-4": [..accuracies..], "5-9": [..accuracies..]}.

    Each list is one accuracy value per Task 2 epoch. Index 0 should be the
    accuracy measured *before* Task 2 training starts (the Task 1 baseline),
    so epoch 0 on the x-axis is the pre-forgetting state.
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )
    plt.figure(figsize=(8, 5))
    for label, accs in history.items():
        epochs = range(len(accs))  # 0 = before Task 2, then per epoch
        plt.plot(epochs, accs, marker="o", label=f"Digits {label}")
    plt.xlabel("Task 2 epoch (0 = before training on 5-9)")
    plt.ylabel("Test Accuracy (\\%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def plot_importance_sweep(histories, task_key, title, save_path):
    """Plot one accuracy curve per EWC importance level.

    histories : dict mapping importance value -> {"0-4": [...], "5-9": [...]}
    task_key  : which set to plot, "0-4" (old) or "5-9" (new)
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )
    plt.figure(figsize=(8, 5))
    for imp in sorted(histories.keys()):
        accs = histories[imp][task_key]
        epochs = range(len(accs))  # 0 = before Task 2, then per epoch
        plt.plot(epochs, accs, marker="o", label=rf"$\lambda = {imp}$")
    plt.xlabel("Task 2 epoch (0 = before training on 5-9)")
    plt.ylabel("Test Accuracy (\\%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--ewc-epochs",
        type=int,
        default=4,
        metavar="N",
        help="epochs for each Task 2 / EWC run (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument(
        "--dry-run", action="store_true", help="quickly check a single pass"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", help="For Saving the current Model"
    )
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("mps")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_accel:
        accel_kwargs = {
            "num_workers": 1,
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST("../data", train=False, transform=transform)

    task1_digits = [0, 1, 2, 3, 4]

    task1_train = filter_by_digits(train_set, task1_digits)
    task1_test = filter_by_digits(test_set, task1_digits)

    task1_train_loader = torch.utils.data.DataLoader(task1_train, **train_kwargs)
    task1_test_loader = torch.utils.data.DataLoader(task1_test, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ------------------------------------------------------------------ #
    # Task 1: train on 0-4
    # ------------------------------------------------------------------ #
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, task1_train_loader, optimizer, epoch)
        test(model, device, task1_test_loader, label="[0-4]")
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_task1.pt")

    # Snapshot the post-Task-1 weights.
    task1_state = copy.deepcopy(model.state_dict())

    # ------------------------------------------------------------------ #
    # Task 3 (naive): train on 5-9 -> catastrophic forgetting
    # ------------------------------------------------------------------ #
    task2_digits = [5, 6, 7, 8, 9]
    task2_train = filter_by_digits(train_set, task2_digits)
    task2_test = filter_by_digits(test_set, task2_digits)
    task2_train_loader = torch.utils.data.DataLoader(task2_train, **train_kwargs)
    task2_test_loader = torch.utils.data.DataLoader(task2_test, **test_kwargs)

    # Make sure we start the naive run from the Task-1 weights.
    model.load_state_dict(task1_state)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("== Baseline before Task 2 ==")
    acc_old = test(model, device, task1_test_loader, label="[0-4]")
    acc_new = test(model, device, task2_test_loader, label="[5-9]")
    history = {"0-4": [acc_old], "5-9": [acc_new]}

    for epoch in range(1, args.ewc_epochs + 1):
        train(args, model, device, task2_train_loader, optimizer, epoch)
        print("-- Training task: 5-9 test --")
        acc_new = test(model, device, task2_test_loader, label="[5-9]")
        print("-- Old task: 0-4 test --")
        acc_old = test(model, device, task1_test_loader, label="[0-4]")
        history["5-9"].append(acc_new)
        history["0-4"].append(acc_old)
        scheduler.step()

    plot_accuracy_history(history, title="Naive Sequential Learning")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_task2.pt")


if __name__ == "__main__":
    main()
