from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from glob import glob
from torchvision.io import decode_image
from pathlib import Path

#--
#CONFIG
batch_size = 64
#--

def map_train_id_to_class_id(dataset, idx):
    """ Mapping the continues training id to dataset class id for specific index, idx """

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    img, label = dataset[idx]
    class_id = idx_to_class[label]
    return class_id


def map_class_id_to_class_label(class_id):
    """ Mapping dataset class_id to corresponding class label """

    # Open dataset mapping txt file
    with open("data/tiny-imagenet-200/words.txt", "r") as f:
        for line in f:
            # Split each line into sections
            parts = line.strip().split("\t")
            # Check if the class_id matches this lines mapping
            if class_id == parts[0]:
                class_label = parts[1]
                break
            else:
                pass
    return class_label


class Tiny_imagnet_testset(Dataset):
    def __init__(self, path, transform):
        # Find all paths to images inside the pathfolder
        self.path = sorted(glob(os.path.join(path, "**", "*.JPEG"), recursive = True))
        self.transform = transform
        self.samples = []

        self.mapping = {}

        # Create mapping for each test image to corresponding class_id
        with open(f'{path + "/val_annotations.txt"}', "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                filename = parts[0]
                class_id = parts[1]
                self.mapping[filename] = class_id

        # Save decoded image and class_id
        for image_path in self.path:
            img = decode_image(image_path)
            filename = Path(image_path).name
            class_id = self.mapping[filename]

            self.samples.append((img, class_id))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():
    """ How to use the functions and methods in this module """
    # Define transforms for the training set
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
    
    # Training set made up of img and train_id
    train_dataset = datasets.ImageFolder(root="data/tiny-imagenet-200/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    # How to use function map_train_id_to_class_id
    print(map_train_id_to_class_id(train_dataset, train_dataset.__getitem__(0)[1]))

    # Test set made up of img and class_id
    test_dataset = Tiny_imagnet_testset("data/tiny-imagenet-200/val", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # How to use map_class_id_to_class_label
    print(map_class_id_to_class_label(test_dataset.__getitem__(0)[1]))

if __name__ == "__main__":
    main()