# make torch tensor data from dataset.




import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta
from utils import local_names

# base year is 2000
# since year 3's Feb 29th is exist
base_year = 2017

def combine_datetime(row:pd.Series):
    year = int(row['year']) + base_year
    date_time = str(year) + '-' + row['date']
    # print(row['date'])

    date = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M')

    return date

def read_pm2(dir: str, local_name: str):
    df = pd.read_csv(os.path.join(dir, local_name + '.csv'))
    df.columns = ['year', 'date', 'loc', 'y']

    df['ds'] = df.apply(combine_datetime, axis=1)

    df.drop(['year', 'loc', 'date'], axis=1, inplace=True)
    df = df.reindex(columns=['ds', 'y'])

    return df

# TODO (use aws data)
# 1. make a list of csv files name (계룡, 공주, 논산, ...)
# 2. concat all aws data of above list (계룡-기온, 계룡-풍향, 계룡-풍속, 계룡-강수량, 계룡-습도, 공주-기온, 공주-풍향, ...)
# 3. seperate data as 5 day period (3 days for train, 2 day for test)
# 4. make torch tensor data from above data
def load_data():
    data = []
    # create input/output sequences
    input_seq_len = 24*3
    output_seq_len = 24*2
    input_seq = []
    output_seq = []
    for i in range(len(data) - input_seq_len - output_seq_len + 1):
        input_seq.append(data[i:i+input_seq_len])
        output_seq.append(data[i+input_seq_len:i+input_seq_len+output_seq_len])
    input_seq = np.array(input_seq)
    output_seq = np.array(output_seq)

    return input_seq, output_seq


if __name__ == "__main__":
    df = read_pm2('dataset/TRAIN', '공주')
    print(df.head())
