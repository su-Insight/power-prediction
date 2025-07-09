# 电力消耗预测项目

## 概述
本项目专注于利用高级机器学习模型预测家庭电力消耗，具体包括LSTM、Transformer以及我设计的自定义Autoformer模型。目标是根据历史数据预测未来的全局有功功率消耗，涵盖短期（90天）和长期（365天）两种预测时段。本项目利用UCI机器学习存储库的“个人家庭电力消耗”数据集，并结合天气数据，构建多变量时间序列预测模型。代码设计用于训练、评估和可视化模型性能，为智能家居应用和电网管理提供能源使用模式的洞察。

## 功能
- **数据处理**：将分钟级数据聚合为每日汇总，包含全局有功功率、电压、电流、子表数据及天气变量（如降水、雾天数）等特征。
- **模型实现**：
  - **LSTM**：经典循环神经网络模型，优化用于短期依赖建模。
  - **Transformer**：利用自注意力机制捕捉复杂长距离依赖。
  - **Autoformer**：结合趋势-季节性分解与Transformer编码的创新模型，增强长期预测精度。
- **训练与评估**：支持多次训练运行，配置超参数，使用均方误差（MSE）和平均绝对误差（MAE）作为评估指标，并提供统计摘要（均值和标准差）。
- **可视化**：生成真实值与预测值对比图，以及MAE和MSE柱状图，用于模型性能分析。

## 安装
1. **克隆仓库**：
   ```bash
   git clone https://github.com/your-username/power-consumption-forecasting.git
   cd power-consumption-forecasting
2. **安装依赖**
   ```bash
   pip install -r requirements.txt
3. **下载数据集**
- 从 https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption 获取UCI数据集
- 从 https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles 获取天气数据
- 注：处理成功的数据集`test_daily.csv`和`train_daily.csv`已放置在`data`目录下

## 使用方法
### 配置
- 编辑 `config.py` 定义路径（`TRAIN_FILE`、`TEST_FILE`、`RESULT_DIR`）、特征（`FEATURES`、`TARGET_FEATURE`）及模型参数（`LSTM_SHORT_TERM_PARAMS` 等）。
- 调整 `NUM_RUNS` 以设置实验运行次数。
### 运行项目
使用所需模型和预测时段执行主脚本：

```bash
python main.py --model lstm --horizon 90
python main.py --model transformer --horizon 365
python main.py --model autoformer --horizon 90

- `--model`：从 `lstm`、`transformer` 或 `autoformer` 中选择。
- `--horizon`：设置为 `90` 表示短期预测，`365` 表示长期预测。
### 输出
- 模型：保存在 `results/` 下的 `.pt` 文件。
- 标准化器：保存在 `results/` 下的 `.pkl` 文件。
- 预测结果：保存在 `results/` 下的 `.npy` 文件。
- 图表：生成于 `plots/` 下的MAE/MSE对比图及真实值与预测值可视化。
- 结果：详细和统计结果保存在 `results/` 下的 `.csv` 文件。
## 代码结构
- `data_processing/data_loading.py`：负责数据加载和预处理。
- `models/`：包含模型定义（`LSTMForecast.py`、`TransformerForecast.py`、`Autoformer.py`）。
- `src/train.py`：实现训练循环，包括验证。
- `src/predict.py`：执行预测并计算评估指标。
- `src/plot.py`：可视化结果并保存对比图。
- `main.py`：协调工作流程，包括模型选择和执行。
- `config.py`：存储配置设置。
## 创新：Autoformer模型
Autoformer模型引入了独特的方法，通过分解层将时间序列数据分离为趋势和季节性成分，使用移动平均技术处理。这种分解后的季节性数据由Transformer编码器处理，趋势部分与编码结果重新组合，增强了长期依赖建模能力。这一设计与传统Transformer直接处理序列的方式不同，通过显式处理趋势-季节性模式，特别适用于受天气等外部因素影响的数据集，如本项目中的电力消耗数据。

## 结果与分析
本项目通过多次运行评估模型，提供MSE和MAE指标，并生成统计摘要（均值和标准差）。LSTM在短期预测中表现出色，Transformer在长时依赖建模中表现有限，而Autoformer凭借其创新设计在长期预测中展现优势。详细结果和可视化图表保存在相应目录下，可进一步分析模型性能。
