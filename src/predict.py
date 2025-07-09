import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import (
    FEATURES, TARGET_FEATURE, RESULT_DIR
)

def make_prediction_and_plot(model, X_test, y_test, scaler, horizon, data_save_path):
    print(f"--- 开始预测与评估: {horizon}天 ---")

    # 1. Set model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test)

    # Convert tensors to numpy arrays for processing
    predictions_numpy_scaled = predictions_scaled.cpu().numpy()
    y_test_numpy_scaled = y_test.cpu().numpy()

    # 2. Inverse transform ALL predictions and actuals to their original scale
    def inverse_transform_batch(scaled_data, scaler_obj):
        data_reshaped = scaled_data.reshape(-1, 1)
        num_features = len(FEATURES)
        target_idx = FEATURES.index(TARGET_FEATURE)
        dummy_array = np.zeros((data_reshaped.shape[0], num_features))
        dummy_array[:, target_idx] = data_reshaped[:, 0]
        unscaled_array = scaler_obj.inverse_transform(dummy_array)
        return unscaled_array[:, target_idx]

    predictions_unscaled = inverse_transform_batch(predictions_numpy_scaled, scaler)
    y_test_unscaled = inverse_transform_batch(y_test_numpy_scaled, scaler)

    # 3. Calculate MSE and MAE on the unscaled predictions
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    print(f"  - 评估结果 (on original-scale data):")
    print(f"    - 均方误差 (MSE): {mse:.4f}")
    print(f"    - 平均绝对误差 (MAE): {mae:.4f}")

    # 4. For plotting, we only need the first sequence from the unscaled arrays
    # first_actual_unscaled = y_test_unscaled[:horizon]
    # first_prediction_unscaled = predictions_unscaled[:horizon]

    # 5. Save predictions and actuals only if data_save_path is provided
    if data_save_path:
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.save(f"{data_save_path.replace('prediction', 'actual')}.npy", y_test_unscaled)
        np.save(f"{data_save_path}.npy", predictions_unscaled)

    return mse, mae