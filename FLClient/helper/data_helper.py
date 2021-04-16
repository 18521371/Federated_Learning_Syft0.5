import numpy as np
import pandas as pd
from loguru import logger
import torch as th
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets

class FashionDataset(Dataset):
    def __init__(self, filehandler, transform=None):
        self.data = pd.read_csv(filehandler)
        self.transform = transform
    
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape(1, 784)
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return len(self.data)

# class FashionDataset(Dataset):
#     def __init__(self, filehandler, transform=None):
#         self.data = pd.read_csv(filehandler)
#         self.transform = transform

#         self.images = self.data.iloc[:, 1:].values.astype(np.uint8).reshape(1, 784)
#         self.labels = self.data.iloc[:, 0]

#         for image in self.images:
#         #if self.transform is not None:
#             image = self.transform(image)   #chuyển sang tensors

#     def __getitem__(self, index):
#         return self.images[index], self.labels[index]
    
#     def __len__(self):
#         return len(self.data)

def TrainLoader(train_csv_path, batch_size) -> DataLoader:
    #train_csv = pd.read_csv(train_csv_path)
    train_set = FashionDataset(train_csv_path, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

def TrainLoaderTest(batch_size) -> DataLoader:
    train_data = th.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )
    return train_data

