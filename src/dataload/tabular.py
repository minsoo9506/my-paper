import numpy as np
import torch
from torch.utils.data import Dataset


class tabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx, :], self.y[idx]


class WeightedtabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray):
        super().__init__()

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # sample with replacement
        idx = np.arange(len(x))
        sampled_idx = np.random.choice(idx, size=len(idx), replace=True, p=weight)
        sampled_idx = np.sort(sampled_idx)

        self.x = self.x[sampled_idx]
        self.y = self.y[sampled_idx]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx, :], self.y[idx]
