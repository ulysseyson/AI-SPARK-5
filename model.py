import torch
import torch.nn as nn

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        output = self.fc(lstm_out[:, -1, :])
        return output

