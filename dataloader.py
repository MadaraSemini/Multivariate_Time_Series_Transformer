import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Read data from CSV
file_path = './dataset/merged_file.csv'
df = pd.read_csv(file_path)

# Convert 'Datetime' column to pandas datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Select columns for training
cols = list(df.columns)[1:]  # Assuming columns 1 to 6 contain the data needed for training
df_for_training = df[cols].astype(float)

# Define a custom PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, n_past=546, n_future=1):
        self.n_past = n_past
        self.n_future = n_future
        self.data = data.values

    def __len__(self):
        return len(self.data) - self.n_past - self.n_future + 1

    def __getitem__(self, idx):
        print(idx)
        idx += self.n_past  # Shift the index to create sequences

        # Extract past and future data based on the index
        past_data = self.data[idx - self.n_past:idx]
        future_data = self.data[idx:idx + self.n_future]

        return {
            "input": torch.tensor(past_data, dtype=torch.float32),
            "target": torch.tensor(future_data, dtype=torch.float32),
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
