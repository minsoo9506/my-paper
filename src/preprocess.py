import pandas as pd
import numpy as np

PATH = "../UCR_Anomaly_FullData/"

# train에 기반하여 mean, std로 normalize
def normalize_data(data_name: str):
    """train에 기반하여 mean, std로 normalize

    Parameters
    ----------
    data_name : str

    Returns
    -------
    normalize_data: pd.DataFrame
    """

    data_path = PATH + data_name
    data_path_split = data_path.split("_")
    train_end_idx = data_path_split[4]

    data = pd.read_csv(data_path)
    train_data = data[:train_end_idx]
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    normalize_data = (data - train_mean) / train_std
    return normalize_data


def split_train_valid_test(data_name: str, train_ratio: float = 0.8):
    """train, valid, test data로 나누기

    Parameters
    ----------
    data_name : str
    train_ratio : float, optional
        validation data ratio, by default 0.8

    Returns
    -------
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    """
    normalized_data = normalize_data(data_name)

    data_path = PATH + data_name
    data_path_split = data_path.split("_")
    train_end_idx = int(data_path_split[4] * train_ratio)
    val_end_idx = data_path_split[4]

    train, valid, test = (
        normalized_data[:train_end_idx],
        normalize_data[train_end_idx:val_end_idx],
        normalize_data[val_end_idx:],
    )

    return train, valid, test
