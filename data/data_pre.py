import pandas as pd
import numpy as np
import pandas as pd


def clean_data(df):
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
df_train_cleaned = clean_data(df_train)
df_test_cleaned = clean_data(df_test)

# 检查清理后的数据
print("训练集清理后信息:")
print(df_train_cleaned.info())
print("\n测试集清理后信息:")
print(df_test_cleaned.info())

# 保存清理后的数据
df_train_cleaned.to_csv('train_cleaned.csv', index=False)
df_test_cleaned.to_csv('test_cleaned.csv', index=False)