# folder안에 있는 파일 이름 리스트로 저장
import os

PATH = './UCR_Anomaly_FullData'
file_list = os.listdir(PATH)

print(file_list)

# 출력 결과 저장
def cal(x, y):
    return x + y, x * y
    
add_result1, mat_result1 = cal(1,1)

import pandas as pd

PATH = './run_results/'
FILE_NAME = 'test.txt'

df = pd.read_csv(PATH + FILE_NAME)

df = df.append(
    {
        'data_name' : 'result1',
        'add' : add_result1,
        'mat' : mat_result1
    },
    ignore_index=True
)

df.to_csv(PATH + FILE_NAME, index=False)

import datetime
now = datetime.datetime.now()

# result를 어떤 식으로 저장할지 고민 필요
# data 마다 result.txt 만들고
    # model, 성능지표들, now, config정보들
    # -> 각 data마다 성능지표들 통계량 계산가능 (mean, std)
    
# 처음 txt 파일 만들때
import os
PATH = './UCR_Anomaly_FullData'
file_list = os.listdir(PATH)
for file_name in file_list:
    df = pd.DataFrame(columns = ['Name' , 'Price', 'Stock']) # 정해진 컬럼 넣고
    df.to_csv(PATH + '/result_' + file_name, index=False)