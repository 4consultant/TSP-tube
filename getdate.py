import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
# 将浮点型numpy数据的输出格式变为：不使用科学计数法，保留小数点后两位


def get_data(file_path):
    df = pd.read_csv(file_path)

    df1 = df.iloc[:, 1:2]
    data_x = np.array(df1)

    df2 = df.iloc[:, 2:3]
    data_y = np.array(df2)

    df3 = np.arange(0, len(df1))
    df4 = df3.reshape(-1, 1)
    index_id = np.array(df4)

    index = np.hstack((data_x, data_y, index_id))

    df3 = df.iloc[0, 0]
    data_r = np.array(df3)

    return index, data_r


def change_size(index, r):
    list_x = index[:, 0]
    list_y = index[:, 1]
    list1 = list_x.flatten()
    list2 = list_y.flatten()
    list1 -= min(list1) - 25
    list2 -= min(list2) - 25
    temp2 = max(list2) - min(list2)
    temp = (864 * 0.8) / temp2

    r *= temp
    list1 *= temp
    list2 *= temp

    lis1 = list1.reshape(-1, 1)
    lis2 = list2.reshape(-1, 1)
    index = np.hstack((lis1, lis2))
    return index, r
