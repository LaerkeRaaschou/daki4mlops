# MLOps Project

Implemented by Anne Ingerslev, Lærke Raaschou & Stoyan Mhihaylov

This repository contains the code that makes up our MLOps implementation: an
end-to-end pipeline built around a ResNet-18 image classifier trained on
Tiny ImageNet-200.

## Repository structure

The code is organized into the following folders:

- **`conf/`** — Hydra configuration files, including the main config and
  subfolders for component-specific configs (optimizer, scheduler, data, etc.).
- **`data/`** — The dataloader, the id-to-label mapping file, and the labels for
  the validation set. This is also the directory where the dataset itself lives.
- **`experiments/`** — Implementations of the different experiments: quantization,
  pruning, the ZeRO optimizer, drift detection, and batch inference. Includes a
  `results/` subfolder and a subfolder for the "unlearning" tasks.
- **`model/`** — The model architecture, plus the code and template for generating
  a model card specific to each trained model. Trained model weights are also
  stored here.
- **`monitoring/`** — Code that sets up Grafana for monitoring model performance
  during deployment.
- **`outputs/`** — Hydra's output directory for resolved run configurations.
- **`unittests/`** — Unit tests for the various modules. Coverage is partial —
  not all code is tested yet.

## Key files in the root directory

- **`.pre-commit-config.yaml`** — The set of pre-commit checks that keep the code
  to a consistent standard before it is pushed to GitHub.
- **`deploy_model.py`** — Serves the trained model in deployment.
- **`drift_detection.py`** — Supplementary deployment code that detects
  data/model drift.
- **`inference.py`** — Runs inference on a trained model.
- **`register_model.py`** — Registers a trained model with MLflow.
- **`test.py`** — Evaluates a final model's ability to classify each class.
- **`train.py`** — The main training script. Supports multi-GPU training via
  data distribution (DDP), AMP, and more.

## Dataset setup
We utilized DVC to have the data live on a server wherefrom it can be pulled using the `tiny-imagenet-200.dvc` or it can be downloaded from [official source](http://cs231n.stanford.edu/tiny-imagenet-200.zip).
The pipeline expects the **Tiny ImageNet-200** dataset unpacked into the `data/`
directory and to have the layout looks like this:

```
data/
└── tiny-imagenet-200/
    ├── train/                     # one subfolder per class (ImageFolder format)
    ├── val/
    │   ├── images/
    │   └── val_annotations.txt
    └── wnids.txt
```

The id-to-label mapping (`mapping_path.json`) is generated automatically on the
first training run and reused afterwards.


## Frameworks used for this project
This project utilized many new frameworks to complete the MLOps pipeline, including but not limited to:
MLflow, Hydra, DVC, Git, Grafana, Avalanche, Jenkins, Docker, GitHub, Weights&Biases

## Setting up a development environment

1. Create a virtual environment:
   ```bash
   python3.11 -m venv .venv
   ```
2. Activate it:
   ```bash
   source .venv/bin/activate
   ```
3. Install the development requirements:
   ```bash
   python -m pip install -r requirements-dev.txt
   ```
4. You're all set!
