import numpy as np
import torch
from torch.utils.data import Dataset


class WindowBasedDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window: int):
        """make window based data

        Parameters
        ----------
        x : np.ndarray
            window 기반으로 만들기 이전 input data
        y : np.ndarray
            window 기반으로 만들기 이전 output data
        window : int
            window size
        """

        super().__init__()

        data_len = len(x) - window + 1
        self.x = np.zeros((data_len, window))
        self.y = np.zeros((data_len, window))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window]
            self.y[idx, :] = y[idx : idx + window]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx, :], self.y[idx, :]
