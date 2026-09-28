from typing import Literal

from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PTDataset

from config import *


class HFDataset(PTDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]


def get_dataloader(data_path, batch_size, data_type: Literal['train', 'test'], shuffle=True, drop_last=True):
    dataset = load_from_disk(data_path)
    dataset = dataset['train'] if isinstance(dataset, DatasetDict) else dataset
    dataset.set_format('torch')
    dataset = HFDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

__all__ = [
    'get_dataloader'
]

if __name__ == "__main__":
    train_dataloader = get_dataloader(
        TRAIN_DATA_PATH, 
        batch_size=1, 
        data_type='train', 
        shuffle=True, 
        drop_last=True
    )
    it = iter(train_dataloader)
    batch = next(it)
    for k, v in batch.items():
        print(k, v)
