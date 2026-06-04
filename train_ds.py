import torch
import os
import datetime
import hydra
import deepspeed
from torchvision import transforms
import torch.distributed as dist
from carbontracker.tracker import CarbonTracker

from model.resnet18 import ResNet18
from data.dataloader import get_dataset, get_loaders


# DeepSpeed launcher sets these env vars
def get_dist_info():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def build_ds_config(cfg):
    # DeepSpeed gets its own config dict (the ZeRO stage lives here). Adam is
    # used so optimizer-state sharding has something to shard (2 states/param)
    config = {
        "train_micro_batch_size_per_gpu": cfg.data.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": cfg.deepspeed.lr},
        },
        "zero_optimization": {"stage": cfg.deepspeed.stage},
        "fp16": {"enabled": cfg.deepspeed.fp16},
    }
    return config


def train_model(model_engine, criterion, dataloader, device, epoch, local_rank):
    model_engine.train()

    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        y_pred = model_engine(x)
        loss = criterion(y_pred, y)

        # DeepSpeed handles zero_grad, scaling and the optimizer step
        model_engine.backward(loss)
        model_engine.step()

        if i % 200 == 0 and local_rank == 0:
            print(
                "Train Epoch: %s, Iteration: %s, Train Loss: %s"
                % (epoch, i, loss.item())
            )

        total_loss += loss.detach()
        num_batches += 1

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / num_batches).item()

    return avg_loss


@torch.no_grad()
def val_model(model_engine, criterion, batches, device):
    # Run on all ranks so sharded (stage 3) forward does not deadlock, then
    # all-reduce the counts and report accuracy from the totals
    model_engine.eval()

    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)
    n_batches = torch.tensor(0.0, device=device)

    for x, y in batches:
        x = x.to(device)
        y = y.to(device)

        y_pred = model_engine(x)
        loss = criterion(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)

        correct += (preds == y).sum()
        total += y.size(0)
        loss_sum += loss.detach()
        n_batches += 1

    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)

    accuracy = (correct / total).item()
    avg_loss = (loss_sum / n_batches).item()
    return avg_loss, accuracy


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    torch.manual_seed(cfg.seed)

    rank, world_size, local_rank = get_dist_info()
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    device = torch.device("cuda", local_rank)

    tracker = None
    if local_rank == 0 and cfg.carbontracker:
        tracker = CarbonTracker(epochs=cfg.trainer.epochs, components="gpu")

    data_path = cfg.data.root

    # Build the model. DeepSpeed places it on the right device during init
    model = ResNet18(num_classes=200)
    criterion = instantiate_loss(cfg)

    ds_config = build_ds_config(cfg)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = get_dataset(
        train_dir=f"{data_path}/train",
        transform=transform_train,
        mapping_path=cfg.data.mapping_path,
    )
    val_dataset = get_dataset(
        train_dir=f"{data_path}/train", transform=transform_val, mapping_path=None
    )

    train_loader, val_loader, sampler = get_loaders(
        train_set=train_dataset,
        val_set=val_dataset,
        batch_size=cfg.data.batch_size,
        val_split_size=cfg.data.val_split,
        seed=cfg.seed,
        world_size=world_size,
        rank=rank,
    )
    print(f"Rank {rank} num_batches = {len(train_loader)}")

    for epoch in range(1, cfg.trainer.epochs + 1):
        if cfg.carbontracker:
            tracker.epoch_start()

        torch.cuda.reset_peak_memory_stats()
        if sampler is not None:
            sampler.set_epoch(epoch)

        start = datetime.datetime.now()
        train_loss = train_model(
            model_engine, criterion, train_loader, device, epoch, local_rank
        )
        torch.cuda.synchronize()
        end = datetime.datetime.now()

        epoch_time = (end - start).total_seconds()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        val_loss, val_acc = val_model(model_engine, criterion, val_loader, device)

        if local_rank == 0:
            print(f"ZeRO stage {cfg.deepspeed.stage}")
            print(f"Epoch {epoch} train time: {epoch_time:.2f} s")
            print(f"Peak VRAM: {peak_mb:.1f} MB")
            print(
                f"Epoch {epoch}. "
                f"Train Loss = {train_loss:.4f}. "
                f"Val Loss = {val_loss:.4f}. "
                f"Val Accuracy = {val_acc:.3f}."
            )

        if cfg.carbontracker:
            tracker.epoch_end()

    if cfg.carbontracker:
        tracker.stop()

    if dist.is_initialized():
        dist.destroy_process_group()


def instantiate_loss(cfg):
    # Small helper so the loss still comes from the hydra config
    from hydra.utils import instantiate

    return instantiate(cfg.loss)


if __name__ == "__main__":
    main()
