import os
from glob import glob
from torch.utils.data import Dataset
import torch
from skimage.transform import resize


class ArtifiscoDataset(Dataset):
    def __init__(self, root_dir, width=None, height=None):
        super(ArtifiscoDataset).__init__()
        self.width = width
        self.height = height
        self.files = sorted(glob(os.path.join(root_dir, '**', '*.pth'), recursive=True))

    def __getitem__(self, index):
        data = torch.load(self.files[index])
        if self.width is not None or self.height is not None:
            self.width = self.width or data['image'].shape[1]
            self.height = self.height or data['image'].shape[2]
            data['image'] = resize(data['image'], (self.width, self.height))
        return data

    def __len__(self):
        return len(self.files)
