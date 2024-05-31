import json
import torch
import os
from collections import namedtuple
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from utils import utils

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dim'
    ]
)

data_path = '../data/'
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)


def load_data(name, path=data_path):
    name = os.path.join(path, name)
    data = np.loadtxt(name, delimiter=',', skiprows=1)
    return data


def read_csv(filename, meta_filename=None, header=True, discrete=None):
    filename = os.path.join(data_path, filename)
    meta_filename = os.path.join(data_path, meta_filename)
    data = pd.read_csv(filename, header='infer' if header else None)

    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name']
            for column in metadata['columns']
            if column['type'] != 'continuous'
        ]
    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]
    else:
        discrete_columns = []

    return data, discrete_columns


def max_min_norm(data):
    for i in range(data.shape[-1]):
        min = np.min(data[:, i])
        max = np.max(data[:, i])
        data[:, i] -= min
        data[:, i] /= (max - min + 1e-6)
    return data


def cross_validation(data, mask=None, fixed=False, test_rate=0.2):
    """
    分割数据集交叉验证
    :param data:
    :param mask:
    :param fixed: 是否固定顺序分割
    :param test_rate:
    :return:
    """
    n = data.shape[0]
    test_n = int(n * test_rate)
    if fixed:
        idx = np.arange(n)
    else:
        idx = np.random.permutation(n)
    test_data = data[idx[:test_n], :]
    train_data = data[idx[test_n:], :]
    if mask is None:
        return train_data, test_data
    else:
        test_m = mask[idx[:test_n], :]
        train_m = mask[idx[test_n:], :]
        return train_data, test_data, train_m, test_m

def preprocess(data, discrete_columns):
    '''
    分离数字列和离散列, 并对离散数据进行one-hot编码
    :param data:
    :param discrete_columns:
    :return:
    '''
    dim_num = 0
    categories = []
    data_num = []
    data_cat = []
    for column_name in data.columns:
        if column_name in discrete_columns:
            # 离散数据
            oe = OrdinalEncoder(dtype='int64')
            oe.fit(data[[column_name]], column_name)
            categories.append(len(set(data[column_name])))
            data_cat.append(oe.transform(data[[column_name]]))
        else:
            # 数字数据
            norm = MinMaxScaler()
            norm.fit(data[[column_name]], column_name)
            data_num.append(norm.transform(data[[column_name]]))
            dim_num += 1
    data_cat = np.concatenate(data_cat, axis=1)
    data_num = np.concatenate(data_num, axis=1).astype(float)
    return data_num, data_cat, dim_num, categories

def split_num_cat(data, decoder):
    data = data.reshape(-1, 16, 4)
    x_hat_num, x_hat_cat = decoder(data)
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.detach().cpu().numpy()
    norm = MinMaxScaler()
    norm.fit(syn_num)

    syn_num = norm.inverse_transform(syn_num)
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    return syn_num, syn_cat


def transform(data, discrete_columns=()):
    """
    数据转化
    :param data:
    :param discrete_columns:
    :return:
    """
    output_dim = 0
    trans_info_list = []
    output_info_list = []
    for column_name in data.columns:
        if column_name in discrete_columns:
            trans_info = _discrete_fit(data[[column_name]])
        else:
            trans_info = _continuous_fit(data[[column_name]])
        output_dim += trans_info.output_dim
        output_info_list.append(trans_info.output_info)
        trans_info_list.append(trans_info)
    if data.shape[0] < 500:
        column_data_list = _synchronous_transform(
            data, trans_info_list
        )
    else:
        column_data_list = _parallel_transform(
            data, trans_info_list
        )
    return np.concatenate(column_data_list, axis=1).astype(float), output_info_list, output_dim

def _continuous_fit(data):
    """
    连续数据使用VGM提取分布类别数，转为：类别内数值+类别的one-hot编码
    :param data:
    :return:
    """
    column_name = data.columns[0]
    gm = ClusterBasedNormalizer(
        missing_value_generation='from_column',
        max_clusters=10, weight_threshold=0.005
    )
    gm.fit(data, column_name)
    num_components = sum(gm.valid_component_indicator)

    return ColumnTransformInfo(
        column_name=column_name, column_type='continuous', transform=gm,
        output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
        output_dim=1 + num_components)


def _discrete_fit(data):
    """
    离散数据转为one-hot编码
    :param data:
    :return:
    """
    column_name = data.columns[0]
    one_hot = OneHotEncoder()
    one_hot.fit(data, column_name)
    num_category = len(one_hot.dummies)
    return ColumnTransformInfo(
        column_name=column_name, column_type='discrete', transform=one_hot,
        output_info=[SpanInfo(num_category, 'softmax')], output_dim=num_category
    )


def _synchronous_transform(raw_data, column_transform_info_list):
    """Take a Pandas DataFrame and transform columns synchronous.

    Outputs a list with Numpy arrays.
    """
    column_data_list = []
    for column_transform_info in column_transform_info_list:
        column_name = column_transform_info.column_name
        data = raw_data[[column_name]]
        if column_transform_info.column_type == 'continuous':
            column_data_list.append(_transform_continuous(column_transform_info, data))
        else:
            column_data_list.append(_transform_discrete(column_transform_info, data))

    return column_data_list


def _parallel_transform(raw_data, column_transform_info_list):
    """Take a Pandas DataFrame and transform columns in parallel.

    Outputs a list with Numpy arrays.
    """
    processes = []
    for column_transform_info in column_transform_info_list:
        column_name = column_transform_info.column_name
        data = raw_data[[column_name]]
        process = None
        if column_transform_info.column_type == 'continuous':
            process = delayed(_transform_continuous)(column_transform_info, data)
        else:
            process = delayed(_transform_discrete)(column_transform_info, data)
        processes.append(process)

    return Parallel(n_jobs=-1)(processes)


def _transform_continuous(column_transform_info, data):
    column_name = data.columns[0]
    flattened_column = data[column_name].to_numpy().flatten()
    data = data.assign(**{column_name: flattened_column})
    gm = column_transform_info.transform
    transformed = gm.transform(data)

    #  Converts the transformed data to the appropriate output format.
    #  The first column (ending in '.normalized') stays the same,
    #  but the lable encoded column (ending in '.component') is one hot encoded.
    output = np.zeros((len(transformed), column_transform_info.output_dim))
    output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
    index = transformed[f'{column_name}.component'].to_numpy().astype(int)
    output[np.arange(index.size), index + 1] = 1.0

    return output


def _transform_discrete(column_transform_info, data):
    ohe = column_transform_info.transform
    return ohe.transform(data).to_numpy()