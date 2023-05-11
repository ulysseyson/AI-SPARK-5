import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, output_dim):
        super(LSTM, self).__init__()
        # self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        # self.lstm2 = nn.LSTM(hidden_dim, input_dim, batch_first = True)
        self.fc1 = nn.Linear(hidden_dim, input_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state in lstm1
        h1_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c1_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        # Initialize hidden state and cell state in lstm1
        h2_0 = torch.zeros(1, x.size(0), self.input_dim).to(x.device)
        c2_0 = torch.zeros(1, x.size(0), self.input_dim).to(x.device)
        
        # x -> x[1:]+out[-1] (input will chages as auto regression)
        # final output tensor shape will be (batch_size, output_seq_len, output_dim)
        output = []
        for i in range(self.output_seq_len):
            out, _ = self.lstm1(x, (h1_0, c1_0))
            # out, _ = self.lstm2(out, (h2_0, c2_0))
            out1 = self.fc1(out[:, -1, :]).unsqueeze(1)
            out2 = self.fc2(out[:, -1, :]).unsqueeze(1)
            # auto_regression_tensor = out[:, -1, :].unsqueeze(1)
            output.append(out2)
            # print(out[:, -1, :].unsqueeze(1).shape)
            # print(x[:, 1:, :].shape)
            # break
            if i == self.output_seq_len - 1:
                break
            x = torch.cat([x[:, 1:, :], out1], 1)
            
        output = torch.cat(output, dim=1)
        # output = self.fc(output)
        
        return output
    
class LSTM_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, output_dim):
        super(LSTM_simple, self).__init__()
        # self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        # self.lstm2 = nn.LSTM(hidden_dim, input_dim, batch_first = True)
        self.fc = nn.Linear(input_dim, output_dim*output_seq_len)

    def forward(self, x):
        # Initialize hidden state and cell state in lstm1
        h1_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c1_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm1(x, (h1_0, c1_0))

        output = self.fc(out)
        
        return output
