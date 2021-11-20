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


# 데이터이름에 따라서 train, test로 나누기
def split_train_test(data):
    pass


# data를 torch, numpy로 바꾸기
def transform_to_torch():
    pass


def tranasform_to_np():
    pass
