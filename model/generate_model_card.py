import sys
import json
import datetime
from pathlib import Path

from omegaconf import OmegaConf
from jinja2 import Template


# Static facts (true for every training of this project)
DATASET_CLASSES = 200
DATASET_TRAIN_IMAGES = 100000  # 500 per class
DATASET_VAL_IMAGES = 10000  # 50 per class

# Preprocessing is fixed in train.py, so it is the same for every run
PREPROCESSING = (
    "Training augmentations: RandomResizedCrop(64, scale=(0.7, 1.0)), "
    "RandomHorizontalFlip, ColorJitter(0.2, 0.2, 0.2, 0.1), ToTensor, "
    "Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)). "
    "Validation/test: Resize((64, 64)), Normalize with the same ImageNet "
    "statistics."
)

# Is hardcoded so should be changed if the training and deployment infrastructure change
COMPUTE_INFRASTRUCTURE = (
    "Trained on the AAU AI-Lab cluster using a Singularity container (PyTorch 25.04)."
)
HARDWARE = "NVIDIA L4 GPU(s), Single- or multi-GPU via PyTorch DDP / DeepSpeed."
SOFTWARE = "PyTorch, Hydra, OmegaConf, scikit-learn, Weights & Biases, MLflow."


def load_metrics(metrics_path):
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# Read and get all configurations used for the run
def build_context(cfg, metrics):
    # Run-specific values pulled from the config that trained this model
    optimizer = OmegaConf.select(cfg, "optimizer._target_", default="unknown")
    lr = OmegaConf.select(cfg, "optimizer.lr", default="unknown")
    momentum = OmegaConf.select(cfg, "optimizer.momentum", default=None)
    weight_decay = OmegaConf.select(cfg, "optimizer.weight_decay", default=None)
    batch_size = OmegaConf.select(cfg, "data.batch_size", default="unknown")
    val_split = OmegaConf.select(cfg, "data.val_split", default="unknown")
    epochs = OmegaConf.select(cfg, "trainer.epochs", default="unknown")
    seed = OmegaConf.select(cfg, "seed", default="unknown")
    amp = OmegaConf.select(cfg, "amp.use", default=False)
    scheduler_use = OmegaConf.select(cfg, "scheduler.use", default=False)

    regime = "fp16 mixed precision (AMP via torch.amp)" if amp else "fp32"

    # Config-driven hyperparameter block
    hp_lines = [
        f"- Optimizer: {optimizer}",
        f"- Learning rate: {lr}",
    ]
    if momentum is not None:
        hp_lines.append(f"- Momentum: {momentum}")
    if weight_decay is not None:
        hp_lines.append(f"- Weight decay: {weight_decay}")
    hp_lines += [
        f"- Batch size (per GPU): {batch_size}",
        f"- Epochs: {epochs}",
        f"- Validation split: {val_split}",
        f"- LR scheduler used: {scheduler_use}",
        f"- Random seed: {seed}",
    ]
    training_hyperparameters = "\n".join(hp_lines)

    # Metrics come from evaluation, not the config
    results_lines = [
        f"- Validation accuracy: {metrics.get('val_acc', '[More Information Needed]')}",
        f"- Validation precision (macro): {metrics.get('val_precision', '[More Information Needed]')}",
        f"- Validation recall (macro): {metrics.get('val_recall', '[More Information Needed]')}",
        f"- Test accuracy: {metrics.get('test_acc', '[More Information Needed]')}",
    ]
    results = "\n".join(results_lines)

    return {
        "card_data": "",
        "direct_use": "Image classification on 64x64 RGB images belonging to the 200 Tiny ImageNet classes.",
        "out_of_scope_use": "Not intended for production use or for images outside the Tiny ImageNet distribution.",
        "preprocessing": PREPROCESSING,
        "training_regime": regime,
        "training_hyperparameters": training_hyperparameters,
        "speeds_sizes_times": (
            f"Trained for {epochs} epochs at per-GPU batch size {batch_size}. "
            "Per-epoch time and peak VRAM are reported in the experiments section "
            "(see D3 results)."
        ),
        "testing_factors": "Evaluation is reported overall and per class (200 classes).",
        "testing_metrics": (
            "Accuracy, macro-averaged precision and macro-averaged recall "
            "(computed with scikit-learn, zero_division=0)."
        ),
        "results": results,
        "results_summary": metrics.get("results_summary", ""),
        "model_examination": "Per-class accuracy and confidence statistics are reported in test_statistics.",
        "compute_infrastructure": COMPUTE_INFRASTRUCTURE,
        "hardware_requirements": HARDWARE,
        "software": SOFTWARE,
    }


def fill_dataset_facts(text):
    # Replace the literal placeholders in the frontmatter / body text
    replacements = {
        "xx classes": f"{DATASET_CLASSES} classes",
        "xxx images": f"{DATASET_TRAIN_IMAGES:,} images",
        "For training the training set from the Tiny-imagenet-200 dataset is used. It contains XXXX.": f"For training, the training set from Tiny ImageNet-200 is used. It contains "
        f"{DATASET_TRAIN_IMAGES:,} images across {DATASET_CLASSES} classes "
        f"(500 per class), 64x64 RGB.",
        "For testing the validation set from Tiny-imagenet-200 dataset is used, containing XXXX.": f"For testing, the validation set from Tiny ImageNet-200 is used, containing "
        f"{DATASET_VAL_IMAGES:,} images ({DATASET_VAL_IMAGES // DATASET_CLASSES} per class).",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main():
    template_path = Path("modelcard_template.md")
    config_path = sys.argv[2]  # e.g. outputs/2026-06-04/15-26-35/.hydra/config.yaml
    metrics_path = sys.argv[3] if len(sys.argv) > 3 else None
    out_path = "model_card.md"

    cfg = OmegaConf.load(config_path)
    metrics = load_metrics(metrics_path)
    context = build_context(cfg, metrics)

    template = Template(Path(template_path).read_text())
    rendered = template.render(**context)
    rendered = fill_dataset_facts(rendered)

    Path(out_path).write_text(rendered)
    print(f"Model card written to {out_path} (generated {datetime.date.today()})")


if __name__ == "__main__":
    main()
