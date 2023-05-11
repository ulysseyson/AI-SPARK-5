import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, output_dim):
        super(LSTM, self).__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))

        # Use the last hidden state of the LSTM to predict output for each time step in the output sequence
        lstm_out = out[:, -1:, :]
        out = self.fc(lstm_out)

        # Generate the rest of the output sequence using the predicted outputs
        output = []
        for i in range(self.output_seq_len - 1):
            out, (h0, c0) = self.lstm(out, (h0, c0))
            out = self.fc(out)
            output.append(out)

        return torch.cat([out] + output, dim=1)
