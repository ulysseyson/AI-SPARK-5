import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pandas as pd

from data import generate_dataloader, Seq2SeqInferDataset
from model import LSTM
from utils import pm2_locs

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 42
torch.manual_seed(42)

batch_size = 32

# TODO: change as calculated
# data shape
feature_dim = 167
output_feature_dim = 17
hidden_dim = 128

# data shape
output_feature_dim = 17
input_seq_len = 24*2 # 2 days
output_seq_len = 24*3 # 3 days

dataloader = generate_dataloader('dataset/processed/flat_fillna_dataset_test.pt', batch_size=batch_size, shuffle=False)

state_dict = torch.load('model.ckpt')
model = LSTM(feature_dim, hidden_dim, output_seq_len, output_feature_dim)
model.load_state_dict(state_dict)

submission = pd.read_csv('dataset/answer_sample.csv')
submission.columns = ['year', 'date', 'loc', 'pm2']
# print(submission['pm2'][10:50])
# submission['pm2'][10:50] =
# exit()

index_base = 0
for batch_input in dataloader:
    batch_output = model(batch_input)
    batch_output = batch_output.reshape(batch_output.shape[0], output_seq_len, output_feature_dim)
    for output in  batch_output:
        output_t = output.transpose(0,1)

        for i, output_t_loc in enumerate(output_t):
            # print(output_t_loc.detach().numpy())

            # exit()
            submission['pm2'][i * 64 * output_seq_len + output_seq_len * index_base:i * 64 * output_seq_len + output_seq_len * index_base + output_seq_len] = output_t_loc.detach().numpy()

        index_base += 1

submission.columns = ['연도','일시','측정소','PM2.5']

submission.to_csv('submission_1.csv', encoding='utf-8')