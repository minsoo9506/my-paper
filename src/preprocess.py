import pandas as pd
import numpy as np

PATH = "../UCR_Anomaly_FullData/"

# train에 기반하여 mean, std로 normalize하고
def normalize(data_name: str):
    data_path = PATH + data_name
    data_path_split = data_path.split("_")
    train_end_idx = data_path_split[4]
    anomaly_start_idx, anomaly_end_idx = data_path_split[5], data_path_split[6]
    
    data = pd.read_csv(data_path)
    
    

# 데이터이름에 따라서 train, test로 나누기
def split_train_test(data):
    pass

# data를 torch, numpy로 바꾸기
def transform_to_torch():
    pass

def tranasform_to_np():
    pass
