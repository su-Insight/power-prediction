# src/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from .config import FEATURES, TARGET_FEATURE


def load_training_data(train_path, horizon, sequence_length=90):
    """
    Loads and prepares the entire dataset for training.
    """
    # 1. Load the entire dataset
    df = pd.read_csv(train_path, parse_dates=['DateTime'], index_col='DateTime')
    df.ffill(inplace=True)
    df = df[FEATURES]

    # 2. Fit scaler on the entire dataset and transform it
    scaler = MinMaxScaler()
    full_data_scaled = scaler.fit_transform(df)

    # 3. Helper function to create sequences
    target_idx = FEATURES.index(TARGET_FEATURE)

    def create_sequences(data, seq_length, h, target_column_index):
        X, y = [], []
        num_samples = len(data) - seq_length - h + 1
        for i in range(num_samples):
            X.append(data[i: i + seq_length])
            y.append(data[i + seq_length: i + seq_length + h, target_column_index])
        return np.array(X), np.array(y)

    # 4. Create sequences from the entire dataset
    X_train, y_train = create_sequences(full_data_scaled, sequence_length, horizon, target_idx)

    # 5. Convert to PyTorch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    print(f"Data loaded for training: {len(X_train)} total samples.")

    # Return data needed for training and the final prediction
    return X_train, y_train, scaler, full_data_scaled