import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.LSTM import LSTMModel
from utils.data_loader import data_loader

# ================================ 日志配置 ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train_log.txt",
    filemode='w'
)

# ================================ 超参数设置 ================================
LOOKBACK_DAYS = 90
PREDICT_DAYS = 90
BATCH_SIZE = 64
EPOCHS = 70
LR = 0.0005

train_df = data_loader("data/train.csv")
test_df = data_loader("data/test.csv")
logging.info("Train and test data loaded successfully.")

# 特征工程
for df in [train_df, test_df]:
    df["sub_metering_remainder"] = (
        df["Global_active_power"] * 1000 / 60
        - (df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"])
    )

feature_cols = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "sub_metering_remainder"
]
target_col = "Global_active_power"

# 清理数据
train_df = train_df.dropna(subset=feature_cols + [target_col])
test_df = test_df.dropna(subset=feature_cols + [target_col])

# 转换为numpy数组
X_train_np = train_df[feature_cols].values
y_train_np = train_df[[target_col]].values
X_test_np = test_df[feature_cols].values
y_test_np = test_df[[target_col]].values

# ================================ 标准化处理 ================================
from sklearn.preprocessing import StandardScaler

# 特征标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_np)  # 拟合训练集并转换
X_test_scaled = scaler_X.transform(X_test_np)        # 用训练集参数转换测试集

# 目标值标准化
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_np)
y_test_scaled = scaler_y.transform(y_test_np)

# ================================ 滑动窗口函数（保持不变） ================================
def create_dataset(X, y, lookback, predict_days):
    Xs, ys = [], []
    for i in range(lookback, len(X) - predict_days):
        Xs.append(X[i - lookback:i])
        ys.append(y[i:i + predict_days].flatten())
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(X_train_scaled, y_train_scaled, LOOKBACK_DAYS, PREDICT_DAYS)
X_test, y_test = create_dataset(X_test_scaled, y_test_scaled, LOOKBACK_DAYS, PREDICT_DAYS)

# ================================ 逆标准化函数 ================================
def inverse_scale_y(scaled_data):
    """专门用于逆标准化目标值"""
    return scaler_y.inverse_transform(scaled_data.reshape(-1, 1)).flatten()

# ================================ 后续流程保持不变 ================================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_size = int(len(train_dataset) * 0.1)
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)

# ================================ 模型训练与评估 ================================
mse_scores, mae_scores = [], []

for seed in range(5):
    logging.info(f"Start training round {seed}")
    torch.manual_seed(seed)
    model = LSTMModel(input_size=X_train.shape[2], output_dim=PREDICT_DAYS)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    patience = 10
    wait = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

        logging.info(f"Round {seed} - Epoch {epoch + 1} - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_model)
    model.eval()

    all_preds, all_targets = [], []
    # ================================ 测试时逆标准化 ================================
    with torch.no_grad():
        all_preds, all_targets = [], []
        for xb, yb in test_loader:
            output = model(xb)
            all_preds.append(output.numpy())
            all_targets.append(yb.numpy())

    # 逆标准化预测结果和真实值
    y_pred_inv = inverse_scale_y(np.vstack(all_preds))  # 使用新的逆标准化函数
    y_test_inv = inverse_scale_y(np.vstack(all_targets))

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    logging.info(f"Round {seed} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    mse_scores.append(mse)
    mae_scores.append(mae)

    if seed == 0:
        plt.figure(figsize=(12, 5))
        plt.plot(y_test_inv[0], label="Actual")
        plt.plot(y_pred_inv[0], label="Predicted")
        plt.title("Prediction vs Actual (First Sample)")
        plt.legend()
        plt.savefig("prediction_vs_actual.png")
        plt.close()

# ================================ 汇总统计输出 ================================
mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)

logging.info(f"MSE Mean: {mse_mean:.4f}, Std: {mse_std:.4f}")
logging.info(f"MAE Mean: {mae_mean:.4f}, Std: {mae_std:.4f}")
