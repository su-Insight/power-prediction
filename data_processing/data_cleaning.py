import pandas as pd


def update_header():
    # 读取 CSV 文件，不指定列名
    df1 = pd.read_csv('test.csv', header=None)
    # 设置列名
    column_names = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    df1.columns = column_names
    # 保存到新的 CSV 文件
    df1.to_csv('test_with_header.csv', index=False)
    # 如果需要覆盖原始文件，请确保已经备份原始文件
    # df1.to_csv('test.csv', index=False)


def aggregate_data():
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


def clean_data():
    def data_cleaning(df):
        # 先转换除DateTime外的所有列
        cols_to_convert = [col for col in df.columns if col != 'DateTime']
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # 然后删除NA
        df_cleaned = df.dropna().copy()  # 明确创建副本

        return df_cleaned

    # 加载数据
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test_with_header.csv')
    print(df_train.info())
    print(df_test.info())
    # 查看每列的缺失值数量
    print(df_train.isnull().sum())
    print(df_test.isnull().sum())
    # 清理数据
    df_train_cleaned = data_cleaning(df_train)
    df_test_cleaned = data_cleaning(df_test)

    # 检查清理后的数据
    print("训练集清理后信息:")
    print(df_train_cleaned.info())
    print("\n测试集清理后信息:")
    print(df_test_cleaned.info())

    # 保存清理后的数据
    df_train_cleaned.to_csv('train_cleaned.csv', index=False)
    df_test_cleaned.to_csv('test_cleaned.csv', index=False)


if __name__ == '__main__':
    # update_header()
    # 数据清洗
    clean_data()
    # 将数据按天聚合
    aggregate_data()