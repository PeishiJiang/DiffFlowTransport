"""Pytorch dataloader."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from jaxtyping import Array


class TimeSeriesDataset(Dataset):

    def __init__(self, dataframe, targets, features, sequence_length=5):
        """The Time series PyTorch dataset.

        Args:
            dataframe (pandas.DataFrame): The pandas dataframe
            targets (List): The target variable(s)
            features (List): The input features
            sequence_length (int, optional): The lookback period. Defaults to 5.
        """

        # save which are the features and target data and sequence length
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

        # load data from dataframe into tensor
        y = torch.tensor(dataframe[self.targets].values).float()
        X = torch.tensor(dataframe[self.features].values).float()

        # remove padding from data and save data in class
        self.y = y[sequence_length:]
        self.X = X[sequence_length:,:]

        # make extra dataseries for padding
        self.yPadding = y[:sequence_length*2]
        self.XPadding = X[:sequence_length*2,:]
        # self.sequence_length = sequence_length

    def __len__(self):
        # return length shape
        return self.X.shape[0]

    def __getitem__(self, i): 
        # select data from the dataset without padding and return data
        if i > self.sequence_length:
            Xr = self.X[i-self.sequence_length:i,:]
            yr = self.y[i-1]
            return Xr, yr

        # select from the padding dataset and return data
        else:
            Xr = self.XPadding[i:i+self.sequence_length,:]
            yr = self.yPadding[i+self.sequence_length - 1]
            return Xr, yr


def make_pytorch_timeseries_dataloader(
    dataframe, targets, features, sequence_length,
    batch_size=10, dl_seed=12, shuffle=True
):
    torch.manual_seed(dl_seed)
    dataset = TimeSeriesDataset(dataframe,
                                targets=targets,
                                features=features,
                                sequence_length=sequence_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader


# class CustomDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         # Return the size of the dataset, which is the length of one of the arrays
#         return len(self.x)

#     def __getitem__(self, idx):
#         # Retrieve and return the corresponding elements from both arrays
#         sample1 = self.x[idx]
#         sample2 = self.y[idx]
#         return sample1, sample2


# def make_pytorch_data_loader(
#     x: Array,
#     y: Array,
#     batch_size: int=10,
#     dl_seed: int=12,
# ):
#     torch.manual_seed(dl_seed)
#     dataset = CustomDataset(np.array(x), np.array(y))
#     dataloader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=True
#     )
#     return dataloader