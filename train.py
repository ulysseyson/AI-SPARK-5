import torch
import torch.nn as nn

from data import load_data
from model import LSTM

# Instantiate the LSTM model
batch_size = 64
feature_dim = 100
local_num = 10
input_dim = (batch_size, 24*3, feature_dim)
output_dim = (batch_size, 24*2, local_num)
hidden_dim = 128

input_seq, output_seq = load_data()
model = LSTM(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_seq)
    loss = criterion(output, output_seq)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))