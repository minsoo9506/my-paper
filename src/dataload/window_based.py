import numpy as np
import torch
from torch.utils.data import Dataset


class WindowBasedDataset(Dataset):
    """make dataset with window based approach

    Parameters
    ----------
    Dataset : [type]
        [description]
    """

    def __init__(
        self,
        data: np.ndarray,
        window: int,
        seq_len: int,
        feature_dim: int = 1,
    ) -> None:
        """[summary]

        Parameters
        ----------
        data : np.ndarray
            [description]
        window : int
            [description]
        seq_len : int
            [description]
        feature_dim : int, optional
            [description], by default 1
        """
        super().__init__()

        # data preprocessing
        _mean = np.mean(data)
        _std = np.std(data)
        data = (data - _mean) / _std
