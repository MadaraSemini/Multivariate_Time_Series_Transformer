from model import build_transformer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader import TimeSeriesDataset
from datapreprocess import load_data

def get_ds():
    # Load and split the data
    file_path = './dataset/AAPL_data_5min_1.csv'
    train_data, val_data = load_data(file_path)

    # print("before",len(train_data))

    # Create instances of the TimeSeriesDataset
    train_dataset = TimeSeriesDataset(train_data)
    val_dataset = TimeSeriesDataset(val_data)

    # print("after",len(train_dataset))
    # print(type(train_dataset))
    # for idx in range(2):
    #     sample = train_dataset[idx]  # Access each sample in the subset
    #     # Process or print the sample data as needed
    #     print(sample)

    batch_size = 8  # Define your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


    # Example usage of DataLoader
    for batch in train_loader:
        inputs = batch['encoder_input']  # Input sequences
        targets = batch['label']  # Target values

        # Your training loop goes here using inputs and targets
        # Replace this example with your model training code
        print("Input shape:", inputs.shape)
        print("Target shape:", targets)
        break  # Break after the first batch for demonstration

get_ds()

def get_model(config, vocab_tgt_len):
    model = build_transformer(vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

