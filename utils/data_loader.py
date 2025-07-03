import pandas as pd

def data_loader(csv_path):
    '''
    处理数据，将分钟数据处理为以天为单位的数据
    :param csv_path:
    :return:
    '''
    # 读取数据
    df = pd.read_csv(csv_path, encoding='iso-8859-1', low_memory=False)
    df = df.dropna()

    # 转换日期格式
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.date  # 只保留年月日

    # 将数值列转换为浮点数（防止后续出错）
    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 按日期聚合
    agg_df = df.groupby("Date").agg({
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "RR": "first",
        "NBJRR1": "first",
        "NBJRR5": "first",
        "NBJRR10": "first",
        "NBJBROU": "first"
    }).reset_index()

    return agg_df

# 示例使用
if __name__ == "__main__":
    daily_df = data_loader()
    print(daily_df.head())