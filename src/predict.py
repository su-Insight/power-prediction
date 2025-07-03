# src/predict.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import (
    FEATURES, TARGET_FEATURE, RESULT_DIR
)


def make_final_forecast(model, full_data_scaled, ground_truth_df, scaler, horizon, sequence_length, plot_save_path):
    """
    Generates a single forecast for the future using the last available data
    and compares it against the ground truth.
    """
    print(f"--- Generating final forecast for {horizon} days ---")

    # 1. Prepare the input for prediction
    # Use the last `sequence_length` days from the historical data
    last_sequence_scaled = full_data_scaled[-sequence_length:]
    X_predict = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0) # Add batch dimension

    # 2. Set model to evaluation mode and make a single prediction
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(X_predict)

    # 3. Inverse transform the prediction
    # Reshape prediction to be 2D [horizon, 1]
    prediction_scaled_numpy = prediction_scaled.cpu().numpy().flatten().reshape(-1, 1)

    # Create a dummy array to perform inverse scaling
    num_features = len(FEATURES)
    target_idx = FEATURES.index(TARGET_FEATURE)
    dummy_array = np.zeros((prediction_scaled_numpy.shape[0], num_features))
    dummy_array[:, target_idx] = prediction_scaled_numpy[:, 0]

    # Inverse transform and get the unscaled prediction
    prediction_unscaled = scaler.inverse_transform(dummy_array)[:, target_idx]

    # 4. Get the ground truth values for the forecast period
    # Ensure the ground truth dataframe is indexed properly
    actual_unscaled = ground_truth_df[TARGET_FEATURE].iloc[:horizon].values

    # Check for length mismatch
    if len(prediction_unscaled) != len(actual_unscaled):
        print(f"Warning: Length mismatch. Prediction: {len(prediction_unscaled)}, Actual: {len(actual_unscaled)}. Truncating to the shorter length.")
        min_len = min(len(prediction_unscaled), len(actual_unscaled))
        prediction_unscaled = prediction_unscaled[:min_len]
        actual_unscaled = actual_unscaled[:min_len]
        horizon = min_len

    # 5. Calculate MSE and MAE on the unscaled data
    mse = mean_squared_error(actual_unscaled, prediction_unscaled)
    mae = mean_absolute_error(actual_unscaled, prediction_unscaled)
    print(f"  - Evaluation on Final Forecast:")
    print(f"    - Mean Squared Error (MSE): {mse:.4f}")
    print(f"    - Mean Absolute Error (MAE): {mae:.4f}")

    # 6. Plot the forecast vs. the ground truth
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))
    plot_index = range(horizon)

    plt.plot(plot_index, actual_unscaled, label='Actual Future Power', color='blue', marker='o', linestyle='-')
    plt.plot(plot_index, prediction_unscaled, label='Predicted Future Power', color='red', marker='x', linestyle='--')
    plt.title(f'Final Power Forecast vs. Actual ({horizon} Days) - MAE: {mae:.2f}', fontsize=16)
    plt.xlabel('Days into the Future', fontsize=12)
    plt.ylabel('Global Active Power', fontsize=12)
    plt.legend()
    plt.grid(True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"Forecast plot saved to: {plot_save_path}")

    return mse, mae