from dataclasses import dataclass


@dataclass
class Data:
    dataset: str
    path: str


@dataclass
class Model:
    arch: str


@dataclass
class TrainingParam:
    batch_size: int
    lr: float
    num_epochs: int
    optim: str
    optim_momentum: float
    optim_weight_decay: float
    scheduler: str
    step_size: int
    step_gamme: float
    loss: str
    label_smooting: float
