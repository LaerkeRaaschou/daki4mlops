"""
Exercise 2: Unlearning with Gradient Ascent

Task 1 - Train a classifier on the full MNIST dataset (0-9) and record the
         baseline accuracy (overall and per class).
Task 2 - Targeted unlearning: pick a digit to forget (default 7) and apply
         gradient ASCENT on its samples (step to INCREASE the loss on that
         class instead of decreasing it).
Task 3 - Evaluate forgetting: accuracy on the forgotten class should drop a
         lot, while accuracy on the remaining digits should stay high.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR

# LaTeX-style fonts for plots
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


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
        return F.log_softmax(x, dim=1)


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


def unlearn_gradient_ascent(args, model, device, forget_loader, optimizer, epoch):
    """One epoch of gradient ASCENT on the forget set.

    Normal training steps to lower the loss; here we step to RAISE it on the
    target class, so the model gets worse at classifying that digit. We do that
    by back-propagating -loss (descent on -loss == ascent on loss).

    Raw ascent on cross-entropy is unbounded and diverges (loss -> inf -> NaN),
    so we (a) stop ascending once the batch is already confused enough
    (loss >= args.ascent_cap) and (b) clip the gradient norm. That keeps the
    forget class collapsing without wrecking the rest of the model."""
    model.train()
    for batch_idx, (data, target) in enumerate(forget_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if loss.item() >= args.ascent_cap:
            continue  # already forgotten this batch; don't push further
        (-loss).backward()  # gradient ascent
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Unlearn Epoch: {} [{}/{}]\tLoss (ascending): {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(forget_loader.dataset),
                    loss.item(),
                )
            )


def unlearn_confuse(args, model, device, forget_loader, optimizer, epoch, forget_class):
    """One epoch of CONFUSE unlearning on the forget set."""
    model.train()
    for batch_idx, (data, target) in enumerate(forget_loader):
        data = data.to(device)
        # random labels in 0..9 excluding the forget class
        wrong = torch.randint(0, 9, (data.size(0),), device=device)
        wrong[wrong >= forget_class] += 1  # skip the forget class
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, wrong)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Confuse Epoch: {} [{}/{}]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(forget_loader.dataset),
                    loss.item(),
                )
            )


def test(model, device, test_loader, label=""):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
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


@torch.no_grad()
def per_class_accuracy(model, device, test_loader, num_classes=10):
    """Return {class: accuracy %} over the given loader."""
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        for c in range(num_classes):
            mask = target == c
            total[c] += mask.sum().item()
            correct[c] += (pred[mask] == c).sum().item()
    return {
        c: (100.0 * correct[c] / total[c] if total[c] else float("nan"))
        for c in range(num_classes)
    }


def print_per_class(acc_by_class, title):
    print(f"\n{title}")
    for c, acc in acc_by_class.items():
        print(f"  class {c}: {acc:5.1f}%")


def filter_by_digits(dataset, digits):
    """Return a Subset of `dataset` containing only the given digit classes."""
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    mask = torch.isin(targets, torch.tensor(list(digits)))
    indices = torch.where(mask)[0].tolist()
    return Subset(dataset, indices)


def plot_unlearning(history, forget_class, method="ascent", save_path="unlearning.png"):
    """history: {"Forgotten class": [...], "Remaining classes": [...]}.
    Index 0 is before unlearning, then one value per unlearning epoch."""
    plt.figure(figsize=(8, 5))
    for label, accs in history.items():
        plt.plot(range(len(accs)), accs, marker="o", label=label)
    plt.xlabel("Unlearning epoch (0 = before unlearning)")
    plt.ylabel("Test accuracy (\\%)")
    plt.title(f"{method.capitalize()} unlearning of class {forget_class}")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def plot_per_class(before, after, forget_class, save_path="per_class.png"):
    """Grouped bars: per-class accuracy before vs after unlearning.
    `before` and `after` are {class: accuracy %} dicts."""
    classes = sorted(before.keys())
    b = [before[c] for c in classes]
    a = [after[c] for c in classes]
    x = np.arange(len(classes))
    w = 0.4
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, b, w, label="Before unlearning")
    plt.bar(x + w / 2, a, w, label="After unlearning")
    # mark the forgotten class
    plt.axvspan(forget_class - 0.5, forget_class + 0.5, color="red", alpha=0.08)
    plt.xticks(x, classes)
    plt.xlabel("Digit class")
    plt.ylabel("Test accuracy (\\%)")
    plt.title(f"Per-class accuracy before vs. after unlearning class {forget_class}")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST Gradient-Ascent Unlearning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="epochs for the initial full-MNIST training",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="Adadelta lr for the initial training"
    )
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--forget-class", type=int, default=7, help="digit to unlearn")
    parser.add_argument(
        "--method",
        choices=["ascent", "confuse"],
        default="confuse",
        help="unlearning method (confuse is more stable)",
    )
    parser.add_argument("--unlearn-epochs", type=int, default=5)
    parser.add_argument(
        "--unlearn-lr",
        type=float,
        default=1e-3,
        help="SGD lr for unlearning (keep small for ascent)",
    )
    parser.add_argument(
        "--ascent-cap",
        type=float,
        default=5.0,
        help="stop ascending a batch once its loss reaches this",
    )
    parser.add_argument(
        "--clip", type=float, default=1.0, help="gradient-norm clip during ascent"
    )
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")
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
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ---- Task 1: train on all digits, record baseline ----
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, label="[all]")
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_full.pt")

    baseline_per_class = per_class_accuracy(model, device, test_loader)
    print_per_class(baseline_per_class, "Baseline per-class accuracy (after training):")

    # ---- Task 2: gradient-ascent unlearning of the target digit ----
    forget = args.forget_class
    remaining = [d for d in range(10) if d != forget]

    forget_train = filter_by_digits(train_set, [forget])  # ascend on these
    forget_test = filter_by_digits(test_set, [forget])  # measure forgetting
    remaining_test = filter_by_digits(test_set, remaining)  # measure collateral

    forget_loader = torch.utils.data.DataLoader(
        forget_train, batch_size=args.batch_size, shuffle=True
    )
    forget_test_loader = torch.utils.data.DataLoader(
        forget_test, batch_size=args.test_batch_size
    )
    remaining_test_loader = torch.utils.data.DataLoader(
        remaining_test, batch_size=args.test_batch_size
    )

    # A SEPARATE optimizer with a small lr -- ascent with the lr=1.0 Adadelta
    # above would wreck the whole model in one step.
    unlearn_optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr)

    acc_forget = test(model, device, forget_test_loader, label=f"[class {forget}]")
    acc_remaining = test(model, device, remaining_test_loader, label="[remaining]")
    history = {
        f"Forgotten class ({forget})": [acc_forget],
        "Remaining classes": [acc_remaining],
    }

    for epoch in range(1, args.unlearn_epochs + 1):
        if args.method == "ascent":
            unlearn_gradient_ascent(
                args, model, device, forget_loader, unlearn_optimizer, epoch
            )
        else:
            unlearn_confuse(
                args, model, device, forget_loader, unlearn_optimizer, epoch, forget
            )
        acc_forget = test(model, device, forget_test_loader, label=f"[class {forget}]")
        acc_remaining = test(model, device, remaining_test_loader, label="[remaining]")
        history[f"Forgotten class ({forget})"].append(acc_forget)
        history["Remaining classes"].append(acc_remaining)

    # ---- Task 3: report ----
    final_per_class = per_class_accuracy(model, device, test_loader)
    print_per_class(
        final_per_class, f"Per-class accuracy after unlearning class {forget}:"
    )
    plot_unlearning(history, forget, method=args.method)
    plot_per_class(baseline_per_class, final_per_class, forget)

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_unlearned_{forget}.pt")


if __name__ == "__main__":
    main()
