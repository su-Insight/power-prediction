import pandas as pd

# 加载清理后的数据
train_df = pd.read_csv('train_cleaned.csv')
test_df = pd.read_csv('test_cleaned.csv')

# 转换DateTime并设置为索引
for df in [train_df, test_df]:
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)

# 计算sub_metering_remainder并添加到DataFrame
for df in [train_df, test_df]:
    # 计算剩余电量 (Global_active_power单位是千瓦时，需要转换为瓦时)
    df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - \
                                  (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])

# 修改聚合函数以包含新的remainder列
def aggregate_daily(df):
    daily_df = df.resample('D').agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',  # 新增列的聚合
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    })
    return daily_df

# 应用聚合
train_daily = aggregate_daily(train_df)
test_daily = aggregate_daily(test_df)

# 保存结果
train_daily.to_csv('train_daily.csv')
test_daily.to_csv('test_daily.csv')

print("Aggregation complete with remainder column. Results saved to 'train_daily.csv' and 'test_daily.csv'.")