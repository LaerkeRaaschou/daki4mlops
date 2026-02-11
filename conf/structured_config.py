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
    optim: str
    num_epochs: int
    lr: float
