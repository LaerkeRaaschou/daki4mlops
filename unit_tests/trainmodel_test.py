import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train


def _setup_case():
    torch.manual_seed(0)

    num_classes = 10
    x = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, num_classes, (8,))

    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, num_classes))
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epoch = 1

    local_rank = 0

    return model, loader, criterion, optimizer, device, epoch, local_rank


def test_train_model_returns_finite_loss(monkeypatch):
    monkeypatch.setattr(train.wandb, "log", lambda payload: None)

    model, loader, criterion, optimizer, epoch, device, local_rank = _setup_case()

    avg_loss = train.train_model(model, criterion, loader, optimizer, epoch, device, local_rank)

    assert isinstance(avg_loss, float)
    assert torch.isfinite(torch.tensor(avg_loss))


def test_train_model_updates_weights(monkeypatch):
    monkeypatch.setattr(train.wandb, "log", lambda payload: None)

    model, loader, criterion, optimizer, epoch, device, local_rank = _setup_case()

    w_before = model[1].weight.detach().clone()

    _ = train.train_model(model, criterion, loader, optimizer, epoch, device, local_rank)

    w_after = model[1].weight.detach()
    assert not torch.allclose(w_before, w_after), "Expected weights to change after training"


def test_train_model_logs_something(monkeypatch):
    logged = []
    monkeypatch.setattr(train.wandb, "log", lambda payload: logged.append(payload))

    model, loader, criterion, optimizer, epoch, device, local_rank = _setup_case()

    _ = train.train_model(model, criterion, loader, optimizer, epoch, device, local_rank)

    assert len(logged) >= 1, "Expected at least one wandb.log call"

    