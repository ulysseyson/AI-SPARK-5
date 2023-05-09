import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from data import generate_dataloader, Seq2SeqDataset
from model import LSTM

if torch.cuda.is_available():
    device = torch.device('cuda')
else: device = torch.device('cpu')

# Set random seed for reproducibility
seed = 42
torch.manual_seed(42)

# Instantiate the LSTM model
batch_size = 32
# TODO: change as calculated
# data shape
feature_dim = 167
output_feature_dim = 17

input_seq_len = 24*2 # 2 days
output_seq_len = 24*3 # 3 days

# input_dim = (batch_size, 24*3, feature_dim)
# output_dim = (batch_size, 24*2, output_feature_dim)
hidden_dim = 128

train_dataloader, val_dataloader, train_size, val_size = generate_dataloader('dataset/processed/flat_fillna_dataset.pt', batch_size=batch_size, shuffle=True, val_split=0.2)

# print(dataloader.dataset[0][0].device)
# exit()
model = LSTM(feature_dim, hidden_dim, output_seq_len, output_feature_dim)
model.to(device)

# Define the loss function and optimizer
# use MAE error
# criterion = F.l1_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for input_seq, output_seq in train_dataloader:
        input_seq = input_seq.to(device)
        output_seq = output_seq.to(device)
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
        
        train_loss += loss.item() * input_seq.size(0)
    
    # average
    train_loss /= train_size
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_seq, output_seq in val_dataloader:
            input_seq = input_seq.to(device)
            output_seq = output_seq.to(device)
            # Forward pass
            y_pred = model(input_seq)
            # print(y_pred[0])
            # print(output_seq[0])
            loss = F.l1_loss(y_pred, output_seq)
            val_loss += loss.item() * input_seq.size(0)
    # average
    val_loss /= val_size
            
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model checkpoint
torch.save(model.state_dict(), 'model2.ckpt')