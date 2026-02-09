import cuda
import torch
import torchvision
from torchvision import transforms

from model.resnet18 import ResNet18
from data.dataloader import Tiny_imagnet_testset, map_class_id_to_class_label

'''
configuration

weights_path = ""
num_classes = 200
batch_size = 32
data_path = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
top_k = 5
seed = 42

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
cuda.benchmark = True
cuda.manual_seed(seed)
torch.manual_seed(seed)
'''

def load_test_data(batch_size, data_path):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
                                    )    
    test_dataset = Tiny_imagnet_testset(data_path, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def initialize_model(num_classes, weights_path):
    model = ResNet18(num_classes)
    weights = torch.load(weights_path)
    model = model.load_state_dict(weights)
    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}')

def main():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)

    weights_path = ""
    num_classes = 200
    batch_size = 32
    model = initialize_model(num_classes, weights_path)

