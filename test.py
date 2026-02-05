import torch
import torchvision
from torchvision import transforms

from model.resnet18 import ResNet18
from data.dataloader import Validation

def load_test_data(batch_size, data_path):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
    test_dataset = Validation(data_path, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def initialize_weights(model, weights):
    model.load_state_dict(weights)
    return model

