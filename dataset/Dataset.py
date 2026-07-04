import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random


# Dataset class definition

class SaliencyDataset(Dataset):
    def __init__(self, image_files, map_files,
                 target_size=(224, 224), split="",
                 transform=None, map_transform=None):
        self.image_files = image_files
        self.map_files = map_files
        self.target_size = (target_size[1], target_size[0])
        self.split = split.lower()
        self.transform = transform if transform else transforms.ToTensor()
        self.map_transform = map_transform if map_transform else transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        split_name = {
            "train": "Train",
            "val": "Val",
            "test": "Test"
        }.get(self.split, "Unknown")

        return f"SaliencyDataset [{split_name}] — {len(self)} samples"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.image_files[idx]).convert("RGB")
        target = Image.open(self.map_files[idx]).convert("L")

        image = image.resize(self.target_size)
        target = target.resize(self.target_size)

        if self.split == "train" and random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        return self.transform(image), self.map_transform(target)