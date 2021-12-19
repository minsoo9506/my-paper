# PATH = "./UCR_Anomaly_FullData/"
# data_name = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"

# data_path = PATH + data_name

# import pandas as pd
# import numpy as np

# df = pd.read_csv(data_path, header=None)

# np_df = np.loadtxt(data_path)
# print(np_df[:4])

# tmp = np.zeros(5)
# tmp[1:2]=2
# print(tmp)

import random
import numpy as np
p = np.array([0.1,0.1,0.2,0.3,0.3])
x = np.zeros((5,5))
idx = np.arange(len(x))
result = np.sort(np.random.choice(idx, size=len(idx) ,replace=True, p=p))
print(result)