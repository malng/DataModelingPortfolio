from collections import namedtuple

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

def single_category_attribute_analysis(
        train:DataFrame, test:DataFrame, attribute:str, attr_enums:set, fillna='-'):
    DataSet = namedtuple('DataSet', ['name', 'data'])
    trainDataset = DataSet('训练集', train[attribute].copy())
    testDataset = DataSet('测试集', test[attribute].copy())
    # 检查空值
    for dataset in (trainDataset, testDataset):
        if set(dataset.data).issuperset({np.nan}):
            dataset_nan_cnt = dataset.data.isnull().sum()
            dataset_len = dataset.data.shape[0]
            nan_ratio = dataset_nan_cnt / dataset_len
            print(f'{dataset.name}包含{dataset_nan_cnt}个空值，'
                  f'空值占{dataset.name}{round(nan_ratio * 100, 2)}%。')
            dataset.data.fillna(fillna, inplace=True)
        else:
            print(f'{dataset.name}不含空值。')
    # 检查异常值
    for dataset in (trainDataset, testDataset):
        if exception_values := set(dataset.data) - set(fillna) - attr_enums:
            print(f'{dataset.name}存在异常值{",".join(map(str, exception_values))}。')
        else:
            print(f'{dataset.name}不存在异常值。')
        if redundant := attr_enums - set(dataset.data):
            print(f'{dataset.name}不存在{",".join(map(str, redundant))}。')
    # 绘图
    train_df, test_df = (ds.data.value_counts().reset_index()
                         for ds in [trainDataset, testDataset])
    train_df['dataSet'], test_df['dataSet'] = 'train', 'test'
    df = pd.concat([train_df, test_df])
    df.columns = [attribute, 'count', 'dataSet']

    ax = sns.catplot(data=df, kind='bar', x=attribute, y='count', hue='dataSet')
    ax.fig.set_size_inches(20,10)    # 设置画布大小
    ax.set_xlabels(fontsize=15)      # 设置x轴标签字号
    ax.set_ylabels(fontsize=15)      # 设置y轴标签字号
    ax.set_xticklabels(fontsize=10)  # 设置x轴字号
    ax.set_yticklabels(fontsize=10)  # 设置y轴字号
    plt.show()