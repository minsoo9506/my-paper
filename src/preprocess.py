import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

PATH = "../UCR_Anomaly_FullData/"


def normalize_data(PATH: str, data_name: str, train_end_idx: int):
    """make normalized data

    Parameters
    ----------
    PATH : str
        [description]
    data_name : str
        [description]
    train_end_idx : int
        [description]

    Returns
    -------
    normalized_data : np.ndarray
        [description]
    """
    data_path = PATH + data_name
    data = np.loadtxt(data_path + ".txt")

    train_data = data[:train_end_idx]

    train_mean = np.mean(train_data)
    train_std = np.std(train_data)

    normalized_data = (data - train_mean) / train_std

    return normalized_data


def make_y_label(data_len: int, data_name: str):
    """make label of data

    Parameters
    ----------
    data_len : int
        total length of data
    data_name : str
        [description]

    Returns
    -------
    y_label : np.ndarray
        0 for normal data, 1 for abnormal data
    """
    y_label = np.zeros(data_len)
    data_name_split = data_name.split("_")
    abnormal_start_idx, abnormal_end_idx = int(data_name_split[-2]), int(
        data_name_split[-1]
    )
    y_label[abnormal_start_idx:abnormal_end_idx] = 1
    return y_label


def split_train_valid_test(PATH: str, data_name: str, train_ratio: float = 0.8):
    """train, valid, test data로 나누기

    Parameters
    ----------
    PATH : str
    data_name : str
    train_ratio : float, optional
        validation data ratio, by default 0.8

    Returns
    -------
    train_x : np.ndarray
    valid_x : np.ndarray
    test_x : np.ndarray
    train_y : np.ndarray
    valid_y : np.ndarray
    test_y : np.ndarray
    """

    data_name_split = data_name.split("_")
    train_end_idx = int(int(data_name_split[-3]) * train_ratio)
    val_end_idx = int(data_name_split[-3])

    normalized_data = normalize_data(PATH, data_name, train_end_idx)

    data_len = len(normalized_data)
    y_label = make_y_label(data_len, data_name)

    train_x, valid_x, test_x = (
        normalized_data[:train_end_idx],
        normalized_data[train_end_idx:val_end_idx],
        normalized_data[val_end_idx:],
    )

    train_y, valid_y, test_y = (
        y_label[:train_end_idx],
        y_label[train_end_idx:val_end_idx],
        y_label[val_end_idx:],
    )

    return train_x, valid_x, test_x, train_y, valid_y, test_y

from typing import Tuple

def normalize_tabular(df: pd.DataFrame, label_name: str = 'label') -> Tuple[np.ndarray, pd.Series]:    
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(label_name, axis=1))
    y = df[label_name]
    return X, y