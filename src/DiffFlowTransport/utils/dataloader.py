"""Pytorch dataloader."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from jaxtyping import Array


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        # Return the size of the dataset, which is the length of one of the arrays
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve and return the corresponding elements from both arrays
        sample1 = self.x[idx]
        sample2 = self.y[idx]
        return sample1, sample2


def make_pytorch_data_loader(
    x: Array,
    y: Array,
    batch_size: int=10,
    dl_seed: int=12,
):
    torch.manual_seed(dl_seed)
    dataset = CustomDataset(np.array(x), np.array(y))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader