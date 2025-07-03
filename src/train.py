# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


def train_model(model, X_train, y_train, horizon, model_save_path, params):
    """
    Trains the model on the provided data without a validation set.
    """
    print(f"--- 开始训练模型 (无验证集): {model.__class__.__name__} for {horizon}天预测 ---")
    train_params = params['train']

    # 1. Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    # 2. Create PyTorch DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)

    # 3. Training Loop
    print("开始模型训练...")
    model.train()
    for epoch in range(train_params['epochs']):
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{train_params['epochs']}, Train Loss: {avg_train_loss:.6f}")

    # 4. Save the final model after all epochs are complete
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n--- 模型训练完成 ---")
    print(f"最终模型已保存至: {model_save_path}")

    return model