"""
Task 3 - Experience Replay (with Avalanche).

Splits MNIST into two experiences (exp0 = digits 0-4, exp1 = digits 5-9) with a
single shared head (class-incremental), then compares four strategies:
    Naive       - no anti-forgetting mechanism (baseline)
    EWC         - avalanche EWCPlugin only
    Replay      - avalanche ReplayPlugin only
    Replay+EWC  - both combined

Install: pip install avalanche-lib==0.6.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

# LaTeX-style fonts for plots
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


class Net(nn.Module):
    """CNN classifier"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def get_exp_acc(results, exp_id):
    """Per-experience test accuracy (%) from an Avalanche eval() result dict."""
    key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{exp_id:03d}"
    return results.get(key, float("nan")) * 100


def run_strategy(name, plugins, benchmark, args, device):
    """Train one strategy across both experiences."""
    print(f"\n=== {name} ===")
    model = Net().to(device)
    strategy = Naive(
        model=model,
        optimizer=Adam(model.parameters(), lr=args.lr),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,
        eval_mb_size=args.test_batch_size,
        device=device,
        plugins=plugins,
        evaluator=EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            loss_metrics(stream=True),
            loggers=[InteractiveLogger()],
        ),
    )

    old_acc, new_acc = [], []
    for experience in benchmark.train_stream:
        strategy.train(experience)
        results = strategy.eval(benchmark.test_stream)
        old_acc.append(get_exp_acc(results, 0))  # digits 0-4
        new_acc.append(get_exp_acc(results, 1))  # digits 5-9
    return {"old": old_acc, "new": new_acc}


def plot_retention(results, save_path="avl_old_task_retention.png"):
    """Old-task (0-4) accuracy after each experience; the drop is forgetting."""
    plt.figure(figsize=(8, 5))
    for name, hist in results.items():
        plt.plot([0, 1], hist["old"], marker="o", label=name)
    plt.xticks([0, 1], ["after exp0 (0-4)", "after exp1 (5-9)"])
    plt.ylabel("Old-task (0-4) test accuracy (\\%)")
    plt.title("Old-task retention across strategies")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def plot_comparison(results, save_path="avl_final_comparison.png"):
    """Grouped bars: final old-task vs new-task accuracy per strategy."""
    names = list(results.keys())
    old_final = [results[n]["old"][-1] for n in names]
    new_final = [results[n]["new"][-1] for n in names]
    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, old_final, w, label="Old task (0-4)")
    plt.bar(x + w / 2, new_final, w, label="New task (5-9)")
    plt.xticks(x, names)
    plt.ylabel("Final test accuracy (\\%)")
    plt.title("After learning both tasks: old vs. new accuracy")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Task 3 with Avalanche (EWC + Replay)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--mem-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # exp0 = {0,1,2,3,4}, exp1 = {5,6,7,8,9}; single shared head.
    benchmark = SplitMNIST(
        n_experiences=2,
        return_task_id=False,
        fixed_class_order=list(range(10)),
        shuffle=False,
        seed=args.seed,
    )

    # Factories -> fresh (stateful) plugins per run.
    configs = {
        "Naive": lambda: [],
        "EWC": lambda: [EWCPlugin(ewc_lambda=args.ewc_lambda)],
        "Replay": lambda: [ReplayPlugin(mem_size=args.mem_size)],
        "Replay+EWC": lambda: [
            ReplayPlugin(mem_size=args.mem_size),
            EWCPlugin(ewc_lambda=args.ewc_lambda),
        ],
    }

    results = {
        name: run_strategy(name, make(), benchmark, args, device)
        for name, make in configs.items()
    }

    print("\n=== FINAL SUMMARY (accuracy %) ===")
    for name, hist in results.items():
        print(
            f"{name:>11}: old(0-4)={hist['old'][-1]:5.1f}   "
            f"new(5-9)={hist['new'][-1]:5.1f}   "
            f"[old after exp0 = {hist['old'][0]:5.1f}]"
        )

    plot_retention(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
