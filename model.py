import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, output_dim):
        super(LSTM, self).__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_seq_len*output_dim)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        output = self.fc(lstm_out[:, -1, :])
        # output = output.view(output.size(0), self.output_seq_len, -1)
        return output

