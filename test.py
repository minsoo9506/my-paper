PATH = "./UCR_Anomaly_FullData/"
data_name = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"

data_path = PATH + data_name

import pandas as pd
import numpy as np

df = pd.read_csv(data_path, header=None)

np_df = np.loadtxt(data_path)
print(np_df[:4])

tmp = np.zeros(5)
tmp[1:2]=2
print(tmp)


