import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(model, X_train, y_train, X_val, y_val, horizon, model_save_path, params):
    print(f"--- 开始训练模型: {model.__class__.__name__} for {horizon}天预测 ---")
    train_params = params['train']
    # 1. Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    # 2. Create PyTorch DataLoaders for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False)

    # 3. Training Loop
    print("开始模型训练...")
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(train_params['epochs']):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{train_params['epochs']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            if model_save_path:  # 仅在 model_save_path 不为空时保存
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(best_model_state, model_save_path)
                print(f"  -> Best model saved to {model_save_path}")

    print(f"--- 模型训练完成 ---")
    # 加载最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model