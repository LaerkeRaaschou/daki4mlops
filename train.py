import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torchvision 
from model import resnet18 as resnet
from data import dataloader
import os
import wandb

from torch.cuda.amp import GradScaler


def train_model(model, dataloader, optimizer, device, num_epochs):
    model.train()
    loss_function = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y, y_pred)
        loss.backward()
        optimizer.step()

    return loss.item



def main():

    # Set variables
    num_epochs = 10
    batch_size = 10
    learning_rate = 1e-2
    loss = 0

    # Start wandb
    wandb.login()

    # Use device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    # Download data
    data_path = ""

    # Define optimizer
    optimizer = optim.SGD(model_parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if __name__ == "__main__":
    main()
