import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from config import FEATURES, TARGET_FEATURE

def load_data(train_path, test_path, horizon,sequence_length=90):

    # 1. 加载训练和测试数据，并进行基础预处理
    train_df = pd.read_csv(train_path, parse_dates=['DateTime'], index_col='DateTime')
    test_df = pd.read_csv(test_path, parse_dates=['DateTime'], index_col='DateTime')
    train_df.ffill(inplace=True)
    test_df.ffill(inplace=True)

    # 仅使用配置文件中定义的特征
    train_df = train_df[FEATURES]
    test_df = test_df[FEATURES]

    # 2. 数据缩放 (对所有特征进行缩放)
    # Scaler在训练数据上拟合，然后应用到训练集和测试集
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # 获取目标特征在列表中的索引，以便在创建y序列时使用
    target_idx = FEATURES.index(TARGET_FEATURE)

    # 3. 内部辅助函数，用于创建时间序列样本
    def create_sequences(data, seq_length, h, target_column_index):
        X, y = [], []
        num_samples = len(data) - seq_length - h + 1
        for i in range(num_samples):
            # 输入特征序列 (X)
            X.append(data[i : i + seq_length])
            # 目标特征序列 (y)
            y.append(data[i + seq_length : i + seq_length + h, target_column_index])
        return np.array(X), np.array(y)

    # 4. 为训练集和测试集创建序列
    X_train, y_train = create_sequences(train_scaled, sequence_length, horizon, target_idx)
    X_test, y_test = create_sequences(test_scaled, sequence_length, horizon, target_idx)

    # 5. 将Numpy数组转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, scaler