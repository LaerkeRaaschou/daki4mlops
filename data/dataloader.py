import os
from glob import glob
from pathlib import Path
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import hydra

batch_size = 64

def map_train_id_to_class_id(dataset, idx):
    """Map the continuous training id to dataset class id for specific index, idx"""

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    img, label = dataset[idx]
    class_id = idx_to_class[label]
    return class_id


def map_class_id_to_class_label(class_id):
    """Map dataset class_id to corresponding class label"""

    # Open dataset mapping txt file
    with open("data/tiny-imagenet-200/words.txt", "r") as file:
        class_label = None
        for line in file:
            # Split each line into sections
            fields = line.strip().split("\t")
            # Check if the class_id matches this lines mapping
            if class_id == fields[0]:
                class_label = fields[1]
                break
    return class_label


class TinyImagenetTestset(Dataset):
    def __init__(self, root, transform, annotations_path):
        # Find all paths to images inside the pathfolder
        self.image_paths = sorted(glob(os.path.join(root, "**", "*.JPEG"), recursive=True))
        self.transform = transform
        self.mapping = {}

        # Create mapping for each test image to corresponding class_id
        with open(annotations_path, "r") as file:
            for line in file:
                fields = line.strip().split("\t")
                filename = fields[0]
                class_id = fields[1]
                self.mapping[filename] = class_id
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Decode path to image
        image_path = self.image_paths[index]
        img = decode_image(image_path)
        # Map filename to class id
        filename = Path(image_path).name
        class_id = self.mapping[filename]

        # Use transforms
        if self.transform:
            img = self.transform(img)
        
        sample = (img, class_id)
        return sample

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    """ How to use the functions and methods in this module """
    # Define transforms for the training set
    transform_train = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                               std=(0.229, 0.224, 0.225))]
                                        )
    
    transform_test = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
                                    )
    
    # Training set made up of img and train_id
    train_dataset = datasets.ImageFolder(root=cfg.data.train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # How to use function map_train_id_to_class_id
    print(map_train_id_to_class_id(train_dataset, train_dataset[0][1]))

    # Test set made up of img and class_id
    test_dataset = TinyImagenetTestset(root=cfg.data.test_dir, transform=transform_test, annotations_path=cfg.data.test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # How to use map_class_id_to_class_label
    print(map_class_id_to_class_label(test_dataset[0][1]))

if __name__ == "__main__":
    main()