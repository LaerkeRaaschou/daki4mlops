import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torchvision 
from .model.resnet18 import ResNet18
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from data.dataloader import *
import os
import wandb



def train_model(model, dataloader, optimizer, device, epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print("Train Epoch: %s, Iteration: %s, Train Loss: %s" % (epoch, i, loss.item()))
            wandb.log({"Train Loss": loss.item()})
    return loss.item()

@torch.no_grad()
def val_model(model, dataloader, device, epoch):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        y_pred = model(x)
        loss = loss_function(y_pred, y)

        if i % 200 == 0:
            print("Val Epoch: %s, Iteration: %s, Val Loss: %s" % (epoch, i, loss.item()))
            wandb.log({"Val Loss": loss.item()})
    return loss.item()



def main():

    # Set variables
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-2
    

    # Data path
    data_path = "data/tiny-imagenet-200"


    # Use device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")


    # Start wandb
    wandb.login()
    wandb.init(project="tiny-imagenet-resnet18")


    # Initialize model 
    model = ResNet18(num_classes=200).to(device)


    # Define optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    best = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, device, epoch)
        val_loss = val_model(model, test_loader, device, epoch)

        scheduler.step()

if __name__ == "__main__":
    main()
