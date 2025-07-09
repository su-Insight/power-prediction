import os.path
import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import csv
from config import RESULT_DIR, NUM_RUNS

bar_width = 0.5
margin = 1  # 控制边距大小


def plot_mae_comparison(results_dict, plot_save_path, time_frame):
    models = ['lstm', 'transformer', 'autoformer']
    model_names = ['LSTM', 'Transformer', 'Autoformer']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    mae_values = {model: results_dict['mae'][f"{time_frame}_{model}"] for model in models}
    avg_mae = {model: np.mean(mae_values[model]) for model in models}
    std_mae = {model: np.std(mae_values[model]) for model in models}

    x = np.arange(len(models))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, [avg_mae[model] for model in models],
                  width=bar_width,
                  yerr=[std_mae[model] for model in models],
                  capsize=5,
                  color=colors,
                  alpha=0.8)

    # 添加数值标签：平均值 ± 标准差
    for i, model in enumerate(models):
        mean = avg_mae[model]
        std = std_mae[model]
        # ax.text(i, mean + std + 5, f"{mean:.2f} ± {std:.2f}",
        #         ha='center', va='bottom', fontsize=10, color='black')

    ax.set_xlim(-margin, len(models) - 1 + margin)
    ax.set_title(f'MAE Comparison for {time_frame.replace("_", " ").title()}', fontsize=16)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(bars, model_names, fontsize=12)

    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"MAE对比图已保存至: {plot_save_path}")

def plot_mse_comparison(results_dict, plot_save_path, time_frame):
    models = ['lstm', 'transformer', 'autoformer']
    model_names = ['LSTM', 'Transformer', 'Autoformer']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    mse_values = {model: results_dict['mse'][f"{time_frame}_{model}"] for model in models}
    avg_mse = {model: np.mean(mse_values[model]) for model in models}
    std_mse = {model: np.std(mse_values[model]) for model in models}

    x = np.arange(len(models))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, [avg_mse[model] for model in models],
                  width=bar_width,
                  yerr=[std_mse[model] for model in models],
                  capsize=5,
                  color=colors,
                  alpha=0.8)

    # 添加数值标签：平均值 ± 标准差
    for i, model in enumerate(models):
        mean = avg_mse[model]
        std = std_mse[model]
        # ax.text(i, mean + std + 500, f"{mean:.2f} ± {std:.2f}",
        #         ha='center', va='bottom', fontsize=10, color='black')

    ax.set_xlim(-margin, len(models) - 1 + margin)
    ax.set_title(f'MSE Comparison for {time_frame.replace("_", " ").title()}', fontsize=16)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(bars, model_names, fontsize=12)

    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"MSE对比图已保存至: {plot_save_path}")

def plot_true_vs_predicted_comparison(actual_data_path, predicted_data_paths, plot_save_path, model_names, horizon):
    """
    绘制三个模型预测未来90天的电力消耗预测值与真实值对比图（一条真实值和三个模型的预测值）

    参数：
        actual_data_path (str): 真实值.npy文件路径
        predicted_data_paths (list): 包含三个模型预测值.npy文件路径的列表
        plot_save_path (str): 图表保存路径
        model_names (list): 模型名称列表
        horizon (int): 预测时间范围（天数）
    """
    # 加载真实值数据
    y_actual = np.load(actual_data_path)

    # 加载预测值数据
    y_predicted = [np.load(pred_path) for pred_path in predicted_data_paths]

    # 确保数据形状一致
    for y_pred in y_predicted:
        assert y_actual.shape == y_pred.shape, "真实值和预测值维度不一致！"

    # 动态获取时间范围
    plot_index = range(horizon)

    # 绘图设置
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))

    # 绘制真实值曲线
    plt.plot(
        plot_index, y_actual[:horizon],
        label='Actual Future Power',
        color='black',
        marker='o',
        linewidth=2,
        markersize=5
    )

    # 绘制每个模型的预测值曲线
    colors = ['blue', 'red', 'green']
    for i, y_pred in enumerate(y_predicted):
        plt.plot(
            plot_index, y_pred[:horizon],
            label=f'Predicted by {model_names[i]}',
            color=colors[i],
            linestyle='--',
            marker='x',
            linewidth=2,
            markersize=5
        )

    # 添加标题和标签
    plt.title(f'Power Consumption Prediction vs Actual ({horizon} Days)', fontsize=16)
    plt.xlabel('Days into the Future', fontsize=14)
    plt.ylabel('Global Active Power (kW)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)

    # 保存图表
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"结果图已保存至: {plot_save_path}")

def plot_true_vs_predicted(actual_data_path, predicted_data_path, plot_save_path):
    """
    绘制真实值 vs 预测值，并计算 MAE 和 MSE

    参数:
        actual_data_path (str): 真实值.npy文件路径
        predicted_data_path (str): 预测值.npy文件路径
        plot_save_path (str): 图表保存路径
    """
    horizon = int(re.search(r'term_(\d+)_days', actual_data_path).group(1))
    # 加载数据
    y_actual = np.load(actual_data_path)
    y_predicted = np.load(predicted_data_path)

    # 确保数据形状一致
    assert y_actual.shape == y_predicted.shape, "真实值和预测值维度不一致！"

    # 计算评估指标
    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)

    print(f"MAE = {mae:.2f} kW")
    print(f"MSE = {mse:.2f} kW²")

    y_actual, y_predicted = y_actual[:horizon], y_predicted[:horizon]

    # 动态获取时间范围
    horizon = y_actual.shape[0]
    plot_index = range(horizon)

    # 绘图设置
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))

    # 绘制曲线
    plt.plot(
        plot_index, y_actual,
        label='Actual Future Power',
        color='blue',
        marker='o',
        linewidth=2,
        markersize=5
    )
    plt.plot(
        plot_index, y_predicted,
        label='Predicted Future Power',
        color='red',
        linestyle='--',
        marker='x',
        linewidth=2,
        markersize=5
    )

    # 添加误差区域
    plt.fill_between(
        plot_index,
        y_actual,
        y_predicted,
        color='gray',
        alpha=0.2,
        label='Absolute Error'
    )

    # 标题和标签（包含MAE和MSE）
    title_text = f"""Power Consumption Prediction vs Actual ({horizon} Days) MAE = {mae:.2f} kW | MSE = {mse:.2f} kW²"""
    plt.title(
        title_text,
        fontsize=16,
        pad=20
    )
    plt.xlabel('Days into the Future', fontsize=14)
    plt.ylabel('Global Active Power (kW)', fontsize=14)

    # 添加指标框
    metrics_text = f"MAE: {mae:.2f} kW\nMSE: {mse:.2f} kW²"
    plt.text(
        0.95, 0.15,
        metrics_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        horizontalalignment='right',
        fontsize=12
    )

    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)

    # 保存图表
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"结果图已保存至: {plot_save_path}")

    return mae, mse


def save_results_to_csv(results_dict, filename):
    """将结果字典保存到CSV文件"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Run', 'MAE', 'MSE'])
        for model, mae_list in results_dict['mae'].items():
            for run, (mae, mse) in enumerate(zip(mae_list, results_dict['mse'][model]), 1):
                writer.writerow([model, run, mae, mse])


def calculate_statistics(results_dict):
    """计算各模型的平均MAE和MSE"""
    stats = {}
    for model in results_dict['mae'].keys():
        avg_mae = np.mean(results_dict['mae'][model])
        std_mae = np.std(results_dict['mae'][model])
        avg_mse = np.mean(results_dict['mse'][model])
        std_mse = np.std(results_dict['mse'][model])
        stats[model] = {
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'avg_mse': avg_mse,
            'std_mse': std_mse
        }
    return stats


def save_statistics_to_csv(stats_dict, filename):
    """将统计结果保存到CSV文件"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Avg MAE', 'MAE Std', 'Avg MSE', 'MSE Std'])
        for model, values in stats_dict.items():
            writer.writerow([
                model,
                f"{values['avg_mae']:.4f}",
                f"{values['std_mae']:.4f}",
                f"{values['avg_mse']:.4f}",
                f"{values['std_mse']:.4f}"
            ])


if __name__ == '__main__':
    plot_save_dir = '../plots'
    results_dir = '../results'
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 初始化结果存储字典
    results = {
        'mae': defaultdict(list),
        'mse': defaultdict(list)
    }

    # 定义模型配置
    model_configs = [
        ('short_term_90_days', 'lstm'),
        ('long_term_365_days', 'lstm'),
        ('short_term_90_days', 'transformer'),
        ('long_term_365_days', 'transformer'),
        ('short_term_90_days', 'autoformer'),
        ('long_term_365_days', 'autoformer')
    ]

    # 运行所有模型
    for time_frame, model_type in model_configs:
        key = f"{time_frame}_{model_type}"
        for i in range(1, NUM_RUNS + 1):
            mae, mse = plot_true_vs_predicted(
                actual_data_path=os.path.join('..', RESULT_DIR, f"actual_{time_frame}_{model_type}_run{i}.npy"),
                predicted_data_path=os.path.join('..', RESULT_DIR, f"prediction_{time_frame}_{model_type}_run{i}.npy"),
                plot_save_path=os.path.join(plot_save_dir, f"true_vs_predicted_{time_frame}_{model_type}_run{i}.png")
            )
            results['mae'][key].append(mae)
            results['mse'][key].append(mse)

    # 保存详细结果
    save_results_to_csv(results, os.path.join(results_dir, 'detailed_results.csv'))

    # 计算并保存统计结果
    stats = calculate_statistics(results)
    save_statistics_to_csv(stats, os.path.join(results_dir, 'statistical_results.csv'))

    # 打印统计结果
    print("\n=== 模型性能统计 ===")
    for model, values in stats.items():
        print(f"{model}:")
        print(f"  MAE: {values['avg_mae']:.2f} ± {values['std_mae']:.2f}")
        print(f"  MSE: {values['avg_mse']:.2f} ± {values['std_mse']:.2f}")

    # 绘制MSE、MAE对比图
    # 绘制90天的MSE对比图
    plot_mse_comparison(results, os.path.join(plot_save_dir, 'mse_comparison_90_days.png'), 'short_term_90_days')

    # 绘制90天的MAE对比图
    plot_mae_comparison(results, os.path.join(plot_save_dir, 'mae_comparison_90_days.png'), 'short_term_90_days')

    # 绘制365天的MSE对比图
    plot_mse_comparison(results, os.path.join(plot_save_dir, 'mse_comparison_365_days.png'), 'long_term_365_days')

    # 绘制365天的MAE对比图
    plot_mae_comparison(results, os.path.join(plot_save_dir, 'mae_comparison_365_days.png'), 'long_term_365_days')

    # # 绘制预测值与真实值对比图
    # actual_data_path_90 = os.path.join('..', RESULT_DIR, 'actual_short_term_90_days_lstm_run1.npy')
    # predicted_data_paths_90 = [
    #     os.path.join('..', RESULT_DIR, 'prediction_short_term_90_days_lstm_run1.npy'),
    #     os.path.join('..', RESULT_DIR, 'prediction_short_term_90_days_transformer_run1.npy'),
    #     os.path.join('..', RESULT_DIR, 'prediction_short_term_90_days_autoformer_run1.npy')
    # ]
    # plot_true_vs_predicted_comparison(actual_data_path_90, predicted_data_paths_90,
    #                                   os.path.join(plot_save_dir, 'true_vs_predicted_comparison_90_days.png'),
    #                                   ['LSTM', 'Transformer', 'Autoformer'], 90)
    #
    # # 绘制365天的预测值与真实值对比图
    # actual_data_path_365 = os.path.join('..', RESULT_DIR, 'actual_long_term_365_days_lstm_run1.npy')
    # predicted_data_paths_365 = [
    #     os.path.join('..', RESULT_DIR, 'prediction_long_term_365_days_lstm_run1.npy'),
    #     os.path.join('..', RESULT_DIR, 'prediction_long_term_365_days_transformer_run1.npy'),
    #     os.path.join('..', RESULT_DIR, 'prediction_long_term_365_days_autoformer_run1.npy')
    # ]
    # plot_true_vs_predicted_comparison(actual_data_path_365, predicted_data_paths_365,
    #                                   os.path.join(plot_save_dir, 'true_vs_predicted_comparison_365_days.png'),
    #                                   ['LSTM', 'Transformer', 'Autoformer'], 365)