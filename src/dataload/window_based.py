import numpy as np
import torch
from torch.utils.data import Dataset
import copy


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


class WeightedWindowBasedDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window: int, weight: np.ndarray):
        """make window based data, sampling with replacement

        Parameters
        ----------
        x : np.ndarray
            window 기반으로 만들기 이전 input data
        y : np.ndarray
            window 기반으로 만들기 이전 output data
        window : int
            window size
        weight : np.ndarray
            sampling시에 사용할 weight
        """

        super().__init__()

        data_len = len(x) - window + 1
        self.x = np.zeros((data_len, window))
        self.y = np.zeros((data_len, window))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window]
            self.y[idx, :] = y[idx : idx + window]

        # sample with replacement
        idx = np.arange(len(self.x))
        sampled_idx = np.random.choice(idx, size=len(idx), replace=True, p=weight)
        sampled_idx = np.sort(sampled_idx)

        self.x = self.x[sampled_idx]
        self.y = self.y[sampled_idx]

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

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


class RatioWindowDateset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        normal_idx: np.ndarray,
        abnormal_idx: np.ndarray,
        window: int,
    ):
        super().__init__()

        data_len = len(x) - window + 1
        self.x = np.zeros((data_len, window))
        self.y = np.zeros((data_len, window))

        for idx in range(data_len):
            self.x[idx, :] = x[idx : idx + window]
            self.y[idx, :] = y[idx : idx + window]

        self.x[abnormal_idx] = copy.deepcopy(self.x[normal_idx])
        self.y[abnormal_idx] = copy.deepcopy(self.y[normal_idx])

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx, :], self.y[idx]
