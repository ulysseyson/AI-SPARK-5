import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from data import generate_dataloader, Seq2SeqDataset
from model import LSTM

# Set random seed for reproducibility
seed = 42
torch.manual_seed(42)

# Instantiate the LSTM model
batch_size = 32
# TODO: change as calculated
feature_dim = 167
output_feature_dim = 17
input_seq_len = 24*3 # 3 days
output_seq_len = 24*2 # 2 days

# input_dim = (batch_size, 24*3, feature_dim)
# output_dim = (batch_size, 24*2, output_feature_dim)
hidden_dim = 128

dataloader = generate_dataloader('dataset/processed/flat_fillna_dataset.pt', batch_size=batch_size, shuffle=True, random_state=seed)

model = LSTM(feature_dim, hidden_dim, output_seq_len, output_feature_dim)

# Define the loss function and optimizer
# use MAE error
# criterion = F.l1_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for input_seq, output_seq in dataloader:
        # Forward pass
        y_pred = model(input_seq)
        # print(y_pred[0])
        # print(output_seq[0])
        loss = F.l1_loss(y_pred, output_seq)
        # print(loss)
        # break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # break
    # Print loss
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')