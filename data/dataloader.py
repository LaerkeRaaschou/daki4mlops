import os
from glob import glob
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

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
        return len(self.image_paths)

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


def map_train_id_to_class_id(dataset, train_id):
    """Map the continuous training id to dataset class id for a specific train_id"""

    idx_to_class = {t: c for c, t in dataset.class_to_idx.items()}
    class_id = idx_to_class[train_id]
    return class_id

def map_class_id_to_class_label(class_id, mapping_file):
    """Map dataset class_id to corresponding class label"""

    # Open dataset mapping txt file
    with open(mapping_file, "r") as file:
        class_label = None
        for line in file:
            # Split each line into sections
            fields = line.strip().split("\t")
            # Check if the class_id matches this lines mapping
            if class_id == fields[0]:
                class_label = fields[1]
                break
    return class_label


def get_train_loader(train_dir, transform_train, batch_size, shuffle):

    # Training set made up of img and train_id
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader

def get_test_loader(test_dir, test_annotations, transform_test, batch_size, shuffle):
    """ How to use the functions and methods in this module """
    # Test set made up of img and class_id
    test_dataset = TinyImagenetTestset(root=test_dir, transform=transform_test, annotations_path=test_annotations)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader