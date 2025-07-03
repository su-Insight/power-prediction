import pandas as pd

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