# main.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from src.data_loader import load_and_split_data
from src.model import LSTMForecast, TransformerForecast, Autoformer
from src.train import train_model
from src.predict import make_final_forecast
from src.config import *


def run_task(model_type, horizon, params, base_model_name, base_scaler_name, base_plot_name, run_num):
    # 1. Create unique filenames for this specific run
    run_model_name = base_model_name.replace('.pt', f'_run{run_num}.pt')
    run_scaler_name = base_scaler_name.replace('.pkl', f'_{model_type}_run{run_num}.pkl')
    run_plot_name = f'{base_plot_name}_{model_type}_final_forecast_run{run_num}.png'

    model_path = os.path.join(MODEL_DIR, run_model_name)
    scaler_path = os.path.join(MODEL_DIR, run_scaler_name)
    plot_path = os.path.join(RESULT_DIR, run_plot_name)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Load and split data for training and validation
    X_train, y_train, X_val, y_val, scaler, full_data_scaled = load_and_split_data(
        train_path=TRAIN_FILE,
        sequence_length=SEQUENCE_LENGTH,
        horizon=horizon
    )
    joblib.dump(scaler, scaler_path)
    print(f"Scaler for run {run_num} saved to {scaler_path}")

    # 3. Create the model instance
    if model_type == 'lstm':
        model = LSTMForecast(output_size=horizon, **params['model'])
    elif model_type == 'transformer':
        model = TransformerForecast(output_size=horizon, **params['model'])
    elif model_type == 'autoformer':
        model = Autoformer(output_size=horizon, **params['model'])
    else:
        raise ValueError("Unknown model type specified.")
    print(f"Initialized {model_type.upper()} model for run {run_num}.")

    # 4. Train the model using the dedicated training and validation sets
    trained_model = train_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        horizon=horizon,
        model_save_path=model_path,
        params=params
    )

    # 5. Make and evaluate the final forecast on unseen test data
    # Load the ground truth data for the forecast period
    ground_truth_df = pd.read_csv(TEST_FILE, parse_dates=['DateTime'], index_col='DateTime')
    ground_truth_df.ffill(inplace=True)

    mse, mae = make_final_forecast(
        model=trained_model,
        full_data_scaled=full_data_scaled,  # Pass the complete scaled historical data
        ground_truth_df=ground_truth_df,  # Pass the actual future data
        scaler=scaler,
        horizon=horizon,
        sequence_length=SEQUENCE_LENGTH,
        plot_save_path=plot_path
    )
    return mse, mae


def main():
    parser = argparse.ArgumentParser(description="Use PyTorch models for power consumption forecasting.")
    parser.add_argument('--model', '-m', type=str, required=True, choices=['lstm', 'transformer', 'autoformer'],
                        help="Model to use for the experiment.")
    parser.add_argument('--horizon', '-H', type=int, required=True, choices=[90, 365], help="Forecast horizon in days.")
    args = parser.parse_args()

    # --- Configuration loading (remains the same) ---
    if args.horizon == 90:
        horizon = SHORT_TERM_HORIZON
        scaler_name = SHORT_TERM_SCALER_NAME
        plot_name = SHORT_TERM_PLOT_NAME
        if args.model == 'lstm':
            params = LSTM_SHORT_TERM_PARAMS
            model_name = LSTM_SHORT_TERM_MODEL_NAME
        elif args.model == 'transformer':
            params = TRANSFORMER_SHORT_TERM_PARAMS
            model_name = TRANSFORMER_SHORT_TERM_MODEL_NAME
        else:  # autoformer
            params = AUTOFORMER_SHORT_TERM_PARAMS
            model_name = AUTOFORMER_SHORT_TERM_MODEL_NAME
    else:  # args.horizon == 365
        horizon = LONG_TERM_HORIZON
        scaler_name = LONG_TERM_SCALER_NAME
        plot_name = LONG_TERM_PLOT_NAME
        if args.model == 'lstm':
            params = LSTM_LONG_TERM_PARAMS
            model_name = LSTM_LONG_TERM_MODEL_NAME
        elif args.model == 'transformer':
            params = TRANSFORMER_LONG_TERM_PARAMS
            model_name = TRANSFORMER_LONG_TERM_MODEL_NAME
        else:  # autoformer
            params = AUTOFORMER_LONG_TERM_PARAMS
            model_name = AUTOFORMER_LONG_TERM_MODEL_NAME

    mse_scores, mae_scores = [], []

    # --- Experiment loop (remains the same) ---
    print("\n" + "=" * 60)
    print(f"Starting {NUM_RUNS} runs for {args.model.upper()} model, {args.horizon}-day forecast...")
    print("=" * 60)

    for i in range(NUM_RUNS):
        print(f"\n--- Experiment Run {i + 1}/{NUM_RUNS} ---")
        mse, mae = run_task(
            model_type=args.model,
            horizon=horizon,
            params=params,
            base_model_name=model_name,
            base_scaler_name=scaler_name,
            base_plot_name=plot_name,
            run_num=i + 1
        )
        if mse is not None and mae is not None:
            mse_scores.append(mse)
            mae_scores.append(mae)

    if mse_scores and mae_scores:
        print("\n" + "=" * 60)
        print(f"Experiment Summary: {args.model.upper()} model, {horizon}-day forecast ({NUM_RUNS} runs average)")
        print("=" * 60)
        print(f"Mean Squared Error (MSE):")
        print(f"  - Average: {np.mean(mse_scores):.4f}")
        print(f"  - Std Dev: {np.std(mse_scores):.4f}")
        print(f"\nMean Absolute Error (MAE):")
        print(f"  - Average: {np.mean(mae_scores):.4f}")
        print(f"  - Std Dev: {np.std(mae_scores):.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()