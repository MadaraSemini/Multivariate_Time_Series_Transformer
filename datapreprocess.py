import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    # Load data from CSV
    df = pd.read_csv(file_path)

    # Convert 'Datetime' column to pandas datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Assuming 'Close' column contains the data needed for training
    df_for_training = df[['Close']].astype(float)

    # Define the percentage split for training and validation
    train_percentage = 0.7
    val_percentage = 0.3

    # Calculate the split index based on the lengths and percentages
    split_index = int(len(df_for_training) * train_percentage)

    # Split the data based on the index
    train_data = df_for_training.iloc[:split_index]
    val_data = df_for_training.iloc[split_index:]

    # Reset the index after splitting
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)

    # Print shapes of training and validation sets
    print("Training set shape:", train_data.shape)
    print("Validation set shape:", val_data.shape)
    
    return train_data, val_data

# # Usage
# file_path = './dataset/AAPL_data_5min_1.csv'
# train_data, val_data = load_data(file_path)

# # Check the shapes of training and validation sets
# print("Training set shape:", train_data.shape)
# print("Validation set shape:", val_data.shape)