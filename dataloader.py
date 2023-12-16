import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, n_past=10, n_future=1):
        self.n_past = n_past
        self.n_future = n_future
        self.data = data.values

    def __len__(self):
        return len(self.data) - self.n_past - self.n_future + 1

    def __getitem__(self, idx):
        idx += self.n_past  # Shift the index to create sequences

        # Extract past and future data based on the index
        enc_input = self.data[idx - self.n_past:idx]
        dec_input = self.data[idx - self.n_past + 2:idx]
        leb = self.data[idx:idx + self.n_future]


        # encoder_input = torch.cat(
        #     [
        #         torch.tensor(enc_input, dtype=torch.float32)
        #     ],
        #     dim=0,
        # )
        # decoder_input = torch.cat(
        #     [
        #         torch.tensor(dec_input, dtype=torch.float32)
        #     ],
        #     dim=0,
        # )

        # label = torch.cat(
        #     [
        #         torch.tensor(leb, dtype=torch.float32)
        #     ],
        #     dim=0,
        # )


        # Convert sequences into tensors and flatten them
        encoder_input = torch.tensor(enc_input, dtype=torch.float32).view(-1)
        decoder_input = torch.tensor(dec_input, dtype=torch.float32).view(-1)
        label = torch.tensor(leb, dtype=torch.float32).view(-1)

        # Concatenate tensors if needed
        # concatenated_tensor = torch.cat([encoder_input, decoder_input, label], dim=0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label
        }

# # Parameters for training
# n_future = 1   # Number of time units to predict in the future
# n_past = 546   # Number of past time units to use for prediction

# # Create dataset and DataLoader
# train_dataset = TimeSeriesDataset(data=df_for_training, n_past=n_past, n_future=n_future)
# batch_size = 64  # Define your batch size
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# # Example usage of DataLoader
# for batch in train_loader:
#     inputs = batch['input']  # Input sequences
#     targets = batch['target']  # Target values

#     # Your training loop goes here using inputs and targets
#     # Replace this example with your model training code
#     print("Input shape:", inputs)
#     print("Target shape:", targets)
#     break  # Break after the first batch for demonstration
