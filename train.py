import torch
from model.resnet18 import ResNet18
from data.dataloader import get_dataset, get_loaders
import wandb
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
import hydra
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
import mlflow
import mlflow.pytorch


# Getting variables nesesary for multi gpu runs
def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# Implementation of early stopping based on no improvement over (delta) within the last (patience) epochs
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


# Train script
def train_model(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch,
    device_type,
    local_rank,
    scaler=None,
):
    model.train()

    # Initialize
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

        if i % 200 == 0 and local_rank == 0:
            print(
                "Train Epoch: %s, Iteration: %s, Train Loss: %s"
                % (epoch, i, loss.item())
            )
            wandb.log({"Train Loss": loss.item()})

        total_loss += loss.detach()
        num_batches += 1

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / num_batches).item()

    if local_rank == 0:
        wandb.log({"Train Avg Loss": avg_loss})
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

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

    mlflow.log_metric("val_loss", avg_loss, step=epoch)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch)
    mlflow.log_metric("val_precision", precision, step=epoch)
    mlflow.log_metric("val_recall", recall, step=epoch)

    return avg_loss, accuracy, precision, recall


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    torch.manual_seed(cfg.seed)

    # Data path
    data_path = cfg.data.root

    # Set up for multiple gpus
    use_ddp = is_distributed()
    if use_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0
        world_size = 1

    if local_rank == 0:
        # Start wandb
        wandb.login()
        wandb.init(project="tiny-imagenet-resnet18")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        mlflow.start_run()

        mlflow.log_params(
            {
                "epochs": cfg.trainer.epochs,
                "batch_size": cfg.data.batch_size,
                "device": cfg.device,
            }
        )

        mlflow.log_dict(dict(cfg), "config.yaml")

    # Initialize model
    model = ResNet18(num_classes=cfg.data.classes).to(device)
    if cfg.compile:
        model = torch.compile(model, backend="eager")

    ddp_model = model
    if use_ddp:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Define optimizer
    optimizer = instantiate(cfg.optimizer, params=ddp_model.parameters())

    scheduler = None
    if cfg.scheduler.use:
        scheduler = instantiate(cfg.scheduler.cfg, optimizer=optimizer)

    criterion = instantiate(cfg.loss)

    scaler = None
    if cfg.amp.use and cfg.device == "cuda":
        scaler = instantiate(cfg.amp.cfg)

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # optional
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

    # Train data and val data = the same
    train_dataset = get_dataset(
        train_dir=f"{data_path}/train",
        transform=transform_train,
        mapping_path=cfg.data.mapping_path,
    )

    val_dataset = get_dataset(
        train_dir=f"{data_path}/train", transform=transform_val, mapping_path=None
    )

    # part into real val and train dataset = non overlapping
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

    if cfg.earlystopping.use:
        early_stopping = EarlyStopping(
            cfg.earlystopping.patience,
            cfg.earlystopping.delta,
            cfg.earlystopping.verbose,
        )

    training_finished = False

    for epoch in range(1, cfg.trainer.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_loss = train_model(
            ddp_model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            cfg.device,
            local_rank,
            scaler,
        )

        if use_ddp:
            dist.barrier()

        if local_rank == 0:
            val_loss, val_acc, val_precision, val_recall = val_model(
                model, criterion, val_loader, device, epoch
            )

            print("Current LR:", optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch}. "
                f"Train Loss = {train_loss:.4f}. "
                f"Val Loss = {val_loss:.4f}. "
                f"Val Accuracy = {val_acc:.3f}. "
                f"Val Precision = {val_precision:.3f}."
                f"Val Recall = {val_recall:.3f}."
            )
            if epoch == 1:
                best = val_acc
                is_best = True
            else:
                is_best = best < val_acc
                if is_best:
                    best = val_acc

            if is_best:
                os.makedirs("artifacts", exist_ok=True)
                model_path = f"artifacts/best_model_epoch{epoch}.pt"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)

            stop = False
            if cfg.earlystopping.use:
                early_stopping.check_early_stop(val_loss)
                stop = early_stopping.stop_training
        else:
            stop = False

        if cfg.scheduler.use:
            scheduler.step()

        if use_ddp:
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()  # all ranks wait until validation/broadcast is done
            stop = bool(stop_tensor.item())

        if stop:
            if local_rank == 0:
                print(f"Early stopping at epoch {epoch}")
                os.makedirs("artifacts", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"artifacts/resnet_18_classifier_epoch{epoch}.pt",
                )
                mlflow.pytorch.log_model(model, artifact_path="final_model")
                mlflow.end_run()
                training_finished = True
            break

    if local_rank == 0 and not training_finished:
        os.makedirs("artifacts", exist_ok=True)
        torch.save(
            model.state_dict(), f"artifacts/resnet_18_classifier_epoch{epoch}.pt"
        )
        mlflow.pytorch.log_model(model, artifact_path="final_model")
        mlflow.end_run()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


"""
transform_train2 = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # optional
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

"""
