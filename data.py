# make torch tensor data from dataset.

import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from datetime import datetime, timedelta
from utils import aws_locs, pm2_locs

from torch.utils.data import Dataset, DataLoader, random_split


use_columns = ['temp', 'wind_dir', 'wind_speed', 'rain', 'humid']
aws_drop_columns = ['year', 'date', 'loc']

# base year is 2000
# since year 3's Feb 29th is exist
base_year = 2017

class Seq2SeqDataset(Dataset):
    def __init__(self, input_seq, output_seq):
        self.input_seq = input_seq
        self.output_seq = output_seq

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_tensor = self.input_seq[idx]
        output_tensor = self.output_seq[idx]

        return input_tensor, output_tensor

class Seq2SeqInferDataset(Dataset):
    def __init__(self, input_seq):
        self.input_seq = input_seq

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_tensor = self.input_seq[idx]

        return input_tensor


def combine_datetime(row:pd.Series):
    year = int(row['year']) + base_year
    date_time = str(year) + '-' + row['date']
    # print(row['date'])

    date = pd.to_datetime(date_time, format='%Y-%m-%d %H:%M')

    return date

def read_pm2(dir: str, pm2_loc: str):
    df = pd.read_csv(os.path.join(dir, pm2_loc + '.csv'))
    df.columns = ['year', 'date', 'loc', f'{pm2_loc}_pm2']
    df['datetime'] = df.apply(combine_datetime, axis=1)
    # drop columns
    df.drop(['year', 'loc', 'date'], axis=1, inplace=True)
    # TODO: fillna
    df.fillna(0, inplace=True)

    return df

def read_aws(dir: str, aws_loc: str):
    df = pd.read_csv(os.path.join(dir, aws_loc + '.csv'))
    df.columns = ['year', 'date', 'loc', f'{aws_loc}_temp', f'{aws_loc}_wind_dir', f'{aws_loc}_wind_speed', f'{aws_loc}_rain', f'{aws_loc}_humid']

    df['datetime'] = df.apply(combine_datetime, axis=1)
    # can select drop columns
    df.drop([x for x in df.columns if x in aws_drop_columns], axis=1, inplace=True)
    # TODO: fillna
    df.fillna(0, inplace=True)

    return df

def generate_save_torch_dataset(dir: str, save: str):
    # if don't give save, return torch dataset
    df = pd.DataFrame()
    for aws_loc in tqdm(aws_locs, desc='load & mere aws data'):
        df_tmp = read_aws(f'{dir}/TRAIN_AWS', aws_loc)
        df = df.join(df_tmp.set_index('datetime'), how='outer')
    for pm2_loc in tqdm(pm2_locs, desc='load & mere pm2 data'):
        df_tmp = read_pm2(f'{dir}/TRAIN', pm2_loc)
        df = df.join(df_tmp.set_index('datetime'), how='outer')

    # create input/output sequences
    input_seq_len = 24*2 # 2 days
    output_seq_len = 24*3 # 3 days
    input_seq = []
    output_seq = []

    # (1,2,3) -> (4,5), (2,3,4) -> (5,6), (3,4,5) -> (6,7) ...
    # may try (1,2,3) -> (4,5), (6,7,8) -> (9,10) ...
    for i in range(len(df) - input_seq_len - output_seq_len + 1):
        input_seq.append(df.iloc[i:i+input_seq_len, :].values)
        output_seq.append(df.iloc[i+input_seq_len:i+input_seq_len+output_seq_len, len(use_columns) * len(aws_locs):].values)

    input_seq = np.array(input_seq)
    output_seq = np.array(output_seq)

    # convert to torch tensor
    input_seq_tensor = torch.from_numpy(input_seq).float()
    output_seq_tensor = torch.from_numpy(output_seq).float()

    output_seq_tensor = output_seq_tensor.reshape(output_seq_tensor.shape[0], -1)

    # save torch dataset
    dataset = Seq2SeqDataset(input_seq_tensor, output_seq_tensor)
    if save is not None:
        torch.save(dataset, save)
    else: return dataset

def generate_dataloader(saved:str, batch_size:int=32, shuffle:bool=True, val_split:float=None):
    dataset = torch.load(saved)
    # split train/val
    if val_split is not None:
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return (DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
                DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle), train_size, val_size)

    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def generate_save_torch_dataset_test(dir: str, save: str):
    # if don't give save, return torch dataset
    df = pd.DataFrame()
    for aws_loc in tqdm(aws_locs, desc='load & mere aws data'):
        df_tmp = read_aws(f'{dir}/TEST_AWS', aws_loc)
        df = df.join(df_tmp.set_index('datetime'), how='outer')
    for pm2_loc in tqdm(pm2_locs, desc='load & mere pm2 data'):
        df_tmp = read_pm2(f'{dir}/TEST_INPUT', pm2_loc)
        df = df.join(df_tmp.set_index('datetime'), how='outer')

    # create input/output sequences
    input_seq_len = 24*2 # 2 days
    output_seq_len = 24*3 # 3 days
    input_seq = []

    for i in range(0, len(df) - input_seq_len - output_seq_len + 1,input_seq_len + output_seq_len):
        input_seq.append(df.iloc[i:i+input_seq_len, :].values)

    input_seq = np.array(input_seq)
    input_seq_tensor = torch.from_numpy(input_seq).float()

    # save torch dataset
    dataset = Seq2SeqInferDataset(input_seq_tensor)
    if save is not None:
        torch.save(dataset, save)
    else: return dataset


if __name__ == "__main__":
    # # test code for generate_save_torch_dataset
    # generate_save_torch_dataset('dataset', save='dataset/processed/flat_fillna_dataset.pt')
    # check dataset
    dataset = torch.load('dataset/processed/flat_fillna_dataset.pt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for input, output in dataloader:
        print(input.shape)
        print(output.shape)
        print(output[0])
        break

    # # test code for generate_save_torch_dataset_test
    # generate_save_torch_dataset_test('dataset', save='dataset/processed/flat_fillna_dataset_test.pt')
    # check dataset
    dataset = torch.load('dataset/processed/flat_fillna_dataset_test.pt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for input in dataloader:
        print(input.shape)
        # print(input)
        break

