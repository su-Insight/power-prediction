import argparse
import itertools
import logging

import numpy as np
import torch
from src.train import train_model
from src.predict import make_prediction_and_plot
from models.Autoformer import Autoformer
from models.LSTMForecast import LSTMForecast
from models.TransformerForecast import TransformerForecast
from data_processing.data_loading import load_data
from config import *

def sprint(msg):
    print(msg)
    logging.info(msg)

def grid_search(model_type, horizon):
    """
    对指定模型类型进行超参数网格搜索，并输出最佳超参数。

    参数:
        model_type (str): 模型类型 ('lstm', 'transformer', 'autoformer')
        horizon (int): 预测时间范围（天数）

    返回:
        dict: 最佳超参数及其性能指标
    """
    # 为每种模型类型定义超参数网格
    param_grids = {
        'lstm': {
            'model': {
                'input_size': [len(FEATURES)],  # 固定
                'hidden_size': [32, 64, 100],  # 3个选项
                'num_layers': [1, 2],  # 2个选项
                'dropout': [0.0, 0.2]  # 2个选项
            },
            'train': {
                'learning_rate': [0.01, 0.001, 0.0005, 0.0001],  # 4个选项
                'epochs': [30, 50, 80],  # 3个选项
                'batch_size': [32]  # 1个选项
            }
        },
        'transformer': {
            'model': {
                'input_size': [len(FEATURES)],  # 固定
                'd_model': [64, 128],  # 2个选项
                'nhead': [4, 8],  # 2个选项
                'num_encoder_layers': [2, 3],  # 2个选项
                'dim_feedforward': [128, 256],  # 2个选项
                'dropout': [0.0, 0.2]  # 2个选项
            },
            'train': {
                'learning_rate': [0.001, 0.0005],  # 2个选项
                'epochs': [30, 60, 80],  # 3个选项
                'batch_size': [32]  # 1个选项
            }
        },
        'autoformer': {
            'model': {
                'input_size': [len(FEATURES)],  # 固定
                'hidden_size': [64, 128],  # 2个选项
                'num_heads': [4, 8],  # 2个选项
                'num_layers': [1, 2],  # 2个选项
                'dropout_rate': [0.0, 0.2],  # 2个选项
                'kernel_size': [15, 25, 35]  # 3个选项
            },
            'train': {
                'learning_rate': [0.01, 0.0005],  # 2个选项
                'epochs': [30, 60],  # 2个选项
                'batch_size': [32]  # 1个选项
            }
        }
    }

    # 获取指定模型类型的参数网格
    if model_type not in param_grids:
        raise ValueError(f"未知模型类型: {model_type}")

    param_grid = param_grids[model_type]
    model_params = param_grid['model']
    train_params = param_grid['train']

    # 生成所有超参数组合
    model_keys = list(model_params.keys())
    train_keys = list(train_params.keys())
    model_values = [model_params[key] for key in model_keys]
    train_values = [train_params[key] for key in train_keys]
    all_combinations = list(itertools.product(*model_values, *train_values))

    # 加载数据
    X_train, y_train, X_test, y_test, scaler = load_data(
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        sequence_length=SEQUENCE_LENGTH,
        horizon=horizon
    )

    best_mse = float('inf')
    best_params = None
    best_mae = None
    results = []

    sprint(f"开始对 {model_type.upper()} 模型进行网格搜索，预测 {horizon} 天...")
    sprint(f"总共超参数组合数: {len(all_combinations)}")

    for idx, combo in enumerate(all_combinations):
        sprint(f"\n--- 网格搜索试验 {idx + 1}/{len(all_combinations)} ---")

        # 创建参数字典
        model_param_dict = {model_keys[i]: combo[i] for i in range(len(model_keys))}
        train_param_dict = {train_keys[i]: combo[i + len(model_keys)] for i in range(len(train_keys))}
        params = {'model': model_param_dict, 'train': train_param_dict}

        # 初始化模型
        if model_type == 'lstm':
            model = LSTMForecast(**params['model'], output_size=horizon)
        elif model_type == 'transformer':
            model = TransformerForecast(**params['model'], output_size=horizon)
        elif model_type == 'autoformer':
            model = Autoformer(**params['model'], output_size=horizon)

        # 训练模型（不保存模型）
        trained_model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            horizon=horizon,
            model_save_path='',  # 传递空路径以避免保存
            params=params
        )

        # 评估模型（不保存预测数据）
        mse, mae = make_prediction_and_plot(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler,
            horizon=horizon,
            data_save_path=''  # 传递空路径以避免保存
        )

        results.append({
            'params': params,
            'mse': mse,
            'mae': mae
        })

        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_mae = mae

        sprint(f"试验 {idx + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        sprint(f"参数: {params}")

    # 打印总结
    sprint("\n" + "=" * 60)
    sprint(f"{model_type.upper()} 模型网格搜索总结，预测 {horizon} 天")
    sprint("=" * 60)
    sprint(f"最佳 MSE: {best_mse:.4f}")
    sprint(f"最佳 MAE: {best_mae:.4f}")
    sprint(f"最佳参数: {best_params}")
    sprint("=" * 60)

    return {
        'best_params': best_params,
        'best_mse': best_mse,
        'best_mae': best_mae,
        'all_results': results
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        filename='grid_search.log',
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # parser = argparse.ArgumentParser(description="对 PyTorch 模型进行超参数网格搜索以预测电力消耗。")
    # parser.add_argument('--model', '-m', type=str, required=True, choices=['lstm', 'transformer', 'autoformer'],
    #                     help="实验使用的模型类型。")
    # parser.add_argument('--horizon', '-H', type=int, required=True, choices=[90, 365], help="预测时间范围（天）。")
    # args = parser.parse_args()

    # 运行网格搜索
    for model_type in ['transformer', 'autoformer', 'lstm']:
        for horizon in [90, 365]:
            result = grid_search(
                model_type=model_type,
                horizon=horizon
            )
            sprint(f"\n{model_type}网格搜索完成！")
            sprint(f"最佳超参数: {result['best_params']}")
            sprint(f"最佳 MSE: {result['best_mse']:.4f}")
            sprint(f"最佳 MAE: {result['best_mae']:.4f}")


if __name__ == '__main__':
    main()