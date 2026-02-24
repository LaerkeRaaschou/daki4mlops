import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model.resnet18 import ResNet18

def initialize_model(num_classes, weights_path):
    model = ResNet18(num_classes)
    if weights_path is None:
        print("No weights provided. Please provide model weights for inference.")
        sys.exit(1)
    
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    return model

def make_dataloader(dir_path, batch_size):
    
    pass
