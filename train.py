import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from data import generate_dataloader, Seq2SeqDataset
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold

from model import LSTM

if torch.cuda.is_available():
    device = torch.device('cuda')
else: device = torch.device('cpu')

print(device)
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

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
hidden_dim = 256

# train_dataloader, val_dataloader, train_size, val_size = generate_dataloader('dataset/processed/flat_fillna_dataset.pt', batch_size=batch_size, shuffle=True, val_split=0.2)

# print(dataloader.dataset[0][0].device)
# exit()
model = LSTM(feature_dim, hidden_dim, output_seq_len, output_feature_dim)
model.to(device)

# Define the loss function and optimizer
# use MAE error
# criterion = F.l1_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = F.l1_loss

# def train(model, optimizer, criterion, dataset_path, num_epochs, model_save_path):

dataloader = generate_dataloader('dataset/processed/flat_fillna_dataset.pt', batch_size=batch_size, shuffle=False)
    
# num_epochs = 10
# for epoch in range(num_epochs):
#     for input_seq, output_seq in dataloader:
#         input_seq = input_seq.to(device)
#         output_seq = output_seq.to(device)
        
#         y_pred = model(input_seq)
#         loss = F.l1_loss(y_pred, output_seq)
#         optimizer.zero_grad()
        
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
        
#         # train_loss += loss.item() * input_seq.size(0)
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

for input_seq, output_seq in dataloader:
    input_seq = input_seq.to(device)
    output = model(input_seq)
    print(output_seq.shape)
    print(output_seq)
    print(output.shape)
    print(output)
    break
exit()

# Save the model checkpoint
torch.save(model.state_dict(), 'result/model/base_model_2e.ckpt')
exit()
            
# Train the model
def train_kfold(model, optimizer, criterion, dataset_path, early_stopping_patience:int=3, k_fold=5, max_epochs=200, model_save_path:str="result/model/best_model.ckpt"):
    # kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    dataset = torch.load(dataset_path)
    fold_loss = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        fold += 1
        train_size = len(train_idx)
        val_size = len(val_idx)
        
        print(f"Fold {fold}/{k_fold} train: {train_size}, val: {val_size}")
        
        tarin_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_dataloader = DataLoader(tarin_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        # val loss patience
        # Track validation loss for early stopping
        best_val_loss = np.Inf
        early_stopping_counter = 0

        for epoch in range(max_epochs):
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
                loss = criterion(y_pred, output_seq)
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
            
            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                # Update best validation loss and save checkpoint
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_save_path}_{fold}.ckpt')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Fold{fold}'s validation loss did not improve for {early_stopping_patience} epochs, stopping early.")
                    print(f"Fold{fold}'s Best Validation Loss: {best_val_loss:.4f}")
                    break
                
        fold_loss.append(best_val_loss)
    print(fold_loss)
    print(f"Average loss: {np.mean(fold_loss):.4f}, std: {np.std(fold_loss):.4f}")

if __name__ == "__main__":
    model = LSTM(feature_dim, hidden_dim, output_seq_len, output_feature_dim)
    model.to(device)
    # train_kfold(model, optimizer, criterion, 'dataset/processed/flat_fillna_dataset.pt', k_fold=5, max_epochs=200, model_save_path="result/model/best_model.ckpt")
    train(model, optimizer, criterion, 'dataset/processed/flat_fillna_dataset.pt', num_epochs=50, model_save_path="result/model/best_model_001.ckpt")