from collections import namedtuple

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


DataSet = namedtuple('DataSet', ['name', 'data'])


def single_category_attribute_analysis(
        train:DataFrame, test:DataFrame, attribute:str, attr_enums:set, fillna='-', size='s'):
    # 创建数据集
    trainDataset = DataSet('训练集', train[attribute].copy())
    testDataset = DataSet('测试集', test[attribute].copy())

    for dataset in (trainDataset, testDataset):
        # 检查空值
        _check_nulls(dataset)
        # 检查异常值
        if exception_values := set(dataset.data) - {np.nan, np.NAN} - attr_enums:
            print(f'{dataset.name}存在异常值{",".join(map(str, exception_values))}。')
        else:
            print(f'{dataset.name}不存在异常值。')
        if redundant := attr_enums - set(dataset.data):
            print(f'{dataset.name}不存在{",".join(map(str, redundant))}。')

    # 准备绘图数据
    train_df, test_df = (ds.data.fillna(fillna).value_counts().reset_index()
                         for ds in [trainDataset, testDataset])
    train_df['dataSet'], test_df['dataSet'] = 'train', 'test'
    df = pd.concat([train_df, test_df])
    df.columns = [attribute, 'count', 'dataSet']
    # 绘图
    ax = sns.catplot(data=df, kind='bar', x=attribute, y='count', hue='dataSet', alpha=0.7)
    if size == 'l':
        ax.fig.set_size_inches(20,10)    # 设置画布大小
        ax.set_xlabels(fontsize=15)      # 设置x轴标签字号
        ax.set_ylabels(fontsize=15)      # 设置y轴标签字号
        ax.set_xticklabels(fontsize=10)  # 设置x轴字号
        ax.set_yticklabels(fontsize=10)  # 设置y轴字号
    plt.show()


def single_numeric_attribute_analysis(train:DataFrame, test:DataFrame, attribute:str, bins=100):
    # 创建数据集
    trainDataset = DataSet('训练集', train[attribute].copy())
    testDataset = DataSet('测试集', test[attribute].copy())

    for dataset in (trainDataset, testDataset):
        # 检查空值
        _check_nulls(dataset)
        # 检查取值范围
        print(f'{dataset.name}的取值范围是{dataset.data.min()} - {dataset.data.max()}。')
    # 准备绘图数据
    train_df, test_df = (ds.data.to_frame() for ds in [trainDataset, testDataset])
    train_df['dataSet'], test_df['dataSet'] = 'train', 'test'
    df = pd.concat([train_df, test_df]).reset_index()
    # 绘图
    sns.histplot(data=df, x=attribute, bins=bins, hue='dataSet', element='step')
    plt.show()


def _check_nulls(dataset):
    if set(dataset.data).issuperset({np.nan}):
        dataset_nan_cnt = dataset.data.isnull().sum()
        dataset_len = dataset.data.shape[0]
        nan_ratio = dataset_nan_cnt / dataset_len
        print(f'{dataset.name}包含{dataset_nan_cnt}个空值，'
              f'空值占{dataset.name}{round(nan_ratio * 100, 2)}%。')
    else:
        print(f'{dataset.name}不含空值。')