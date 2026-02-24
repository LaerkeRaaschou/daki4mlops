import torch
from torch import nn

import train


def test_val_model_perfect_predictions(monkeypatch):
    logged = []
    monkeypatch.setattr(train.wandb, "log", lambda payload: logged.append(payload))

    device = torch.device("cpu")
    num_classes = 3

    y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    x = torch.zeros(len(y), 3, 64, 64)
    x[:, 0, 0, 0] = y.float()

    batches = [(x, y)]

    class ReadLabel(nn.Module):
        def forward(self, x):
            labels = x[:, 0, 0, 0].long()
            logits = torch.full((x.size(0), num_classes), -10.0)
            logits[torch.arange(x.size(0)), labels] = 10.0
            return logits

    model = ReadLabel().to(device)
    criterion = nn.CrossEntropyLoss()

    avg_loss, acc, prec, rec = train.val_model(
        model=model,
        criterion=criterion,
        batches=batches,
        device=device,
        epoch=1,
        num_classes=num_classes,
    )

    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert torch.isfinite(torch.tensor(avg_loss))
    assert len(logged) >= 1


def test_val_model_returns_valid_scalars(monkeypatch):
    logged = []
    monkeypatch.setattr(train.wandb, "log", lambda payload: logged.append(payload))

    device = torch.device("cpu")
    num_classes = 3

    y = torch.tensor([0, 1, 2, 0, 1, 2])
    x = torch.randn(6, 3, 64, 64)

    batches = [(x, y)]

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, num_classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    avg_loss, acc, prec, rec = train.val_model(
        model=model,
        criterion=criterion,
        batches=batches,
        device=device,
        epoch=1,
        num_classes=num_classes,
    )

    for value in [avg_loss, acc, prec, rec]:
        assert isinstance(value, (float, int))
        assert torch.isfinite(torch.tensor(float(value)))

    assert len(logged) >= 1
