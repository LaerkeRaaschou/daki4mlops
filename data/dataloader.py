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

def trainclass_to_label(dataset):
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    image, label = dataset[0]
    class_id = idx_to_class[label]
    print(label)
    print(class_id)
    return class_id

def class_id_to_human_label():
    mapping = {}
    with open("/Users/laerkeraaschou/Desktop/semester4/mlops/daki4mlops/data/tiny-imagenet-200/words.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            class_id = parts[0]
            human_label = parts[1]
            mapping[class_id] = human_label
    return mapping

class Validation(Dataset):
    def __init__(self, path, transform):
        self.path = sorted(glob(os.path.join(path, "**", "*.JPEG"), recursive = True))
        self.transform = transform
        self.samples = []

        self.mapping = {}

        with open(f'{path + "/val_annotations.txt"}', "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                filename = parts[0]
                class_id = parts[1]
                self.mapping[filename] = class_id

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
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder(root="/Users/laerkeraaschou/Desktop/semester4/mlops/daki4mlops/data/tiny-imagenet-200/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffel = True)

    val_dataset = Validation("/Users/laerkeraaschou/Desktop/semester4/mlops/daki4mlops/data/tiny-imagenet-200/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffel = False)


if __name__ == "__main__":
    main()