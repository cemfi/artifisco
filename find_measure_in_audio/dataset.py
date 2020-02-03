import os
from torch.utils.data import Dataset
import torch
import random
import math
from pathlib import Path
import json
from torchvision import transforms


class ArtifiscoDataset(Dataset):
    def __init__(self, root, train_val_split=0.8, train=False):
        # with open(os.path.join(root, 'metadata.json')) as fp:
        #     metadata = json.load(fp)

        self.files = list(Path(root).rglob('*.pth'))
        shuffle = random.Random(42).shuffle  # Make sure to have reproducible shuffling
        shuffle(self.files)

        split_point = math.floor(len(self.files) * train_val_split)
        if train:
            self.files = self.files[:split_point]
        else:
            self.files = self.files[split_point:]

        # self.image_transforms = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((metadata['width_images'], metadata['height_images'])),
        #     transforms.ToTensor(),
        #     transforms.Normalize(metadata['mean_images'], metadata['std_images'])
        # ])
        #
        # self.spectrum_transforms = transforms.Normalize(metadata['mean_spectrums'], metadata['std_spectrums'])

    def __getitem__(self, index):
        data = torch.load(self.files[index])

        data['image'] = torch.stack([
            data['image'],
            data['image'],
            data['image']
        ], dim=0)

        data['spectrum'] = torch.stack([
            data['spectrum'],
            data['spectrum'],
            data['spectrum']
        ], dim=0)

        # data['image'] = self.image_transforms(data['image'])
        # data['spectrum'] = self.spectrum_transforms(data['spectrum'])
        return data

    def __len__(self):
        return len(self.files)
