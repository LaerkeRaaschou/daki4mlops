import os
import json
from glob import glob
from pathlib import Path
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
import torch
from torch.utils.data.distributed import DistributedSampler


class TinyImagenetTestset(Dataset):
    def __init__(self, root, transform, annotations_path, mapping_dict):
        # Find all paths to images inside the pathfolder
        self.image_paths = sorted(
            glob(os.path.join(root, "**", "*.JPEG"), recursive=True)
        )
        self.transform = transform
        self.mapping = {}

        # Create mapping for each test image to corresponding class_id
        with open(annotations_path, "r") as file:
            for line in file:
                fields = line.strip().split("\t")
                filename = fields[0]
                class_id = fields[1]
                self.mapping[filename] = class_id

        self.filename_to_train_id = {}
        for filename, class_id in self.mapping.items():
            if class_id not in mapping_dict:
                raise ValueError(
                    f"Class ID {class_id} not found in mapping dictionary."
                )
            self.filename_to_train_id[filename] = int(mapping_dict[class_id])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Decode path to image
        image_path = self.image_paths[index]
        img = decode_image(image_path)
        # Map filename to train id
        filename = Path(image_path).name
        train_id = self.filename_to_train_id[filename]

        # Use transforms
        if self.transform:
            img = self.transform(img)

        sample = (img, train_id)
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


def get_dataset(train_dir, transform, mapping_path):
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    if mapping_path is not None:
        mapping_path = Path(mapping_path)

        if not mapping_path.exists():
            mapping_path.parent.mkdir(parents=True, exist_ok=True)

            with open(mapping_path, "w") as f:
                json.dump(dataset.class_to_idx, f, indent=2, sort_keys=True)

    return dataset


def get_loaders(
    train_set, val_set, batch_size, val_split_size, seed, world_size=None, rank=None
):
    n = len(train_set)
    val_size = round(n * val_split_size)
    train_size = n - val_size

    # Create random reproducible shuffle of idx
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    # Take idx for train and val set
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    # Create subset of random datasplit
    train_dataset = Subset(train_set, train_idx)

    sampler = None
    if world_size is not None and rank is not None:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    val_dataset = Subset(val_set, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, sampler


def get_test_loader(
    mapping_path, test_dir, test_annotations, transform_test, batch_size, shuffle
):
    """How to use the functions and methods in this module"""
    # Test set made up of img and class_id
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        mapping_dict = json.load(f)

    test_dataset = TinyImagenetTestset(
        root=test_dir,
        transform=transform_test,
        annotations_path=test_annotations,
        mapping_dict=mapping_dict,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader
