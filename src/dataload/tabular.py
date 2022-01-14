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
        """[summary]

        Parameters
        ----------
        idx : int
            [description]

        Returns
        -------
        self.x[idx, :] : torch.tensor
        self.y[idx, :] : torch.tensor
        """
        return self.x[idx, :], self.y[idx, :]