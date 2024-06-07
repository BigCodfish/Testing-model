import os

import numpy as np
import pandas as pd

from datautils.data_transformer import SpanInfo
from utils.painter import draw_sub, draw

result_path = '../result/'
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_path)


class Logger():
    def __init__(self, keys):
        records = {}
        for i in keys:
            records[i] = []
        self.keys = keys
        self.records = records
        self.last_v = None

    def append(self, value):
        self.last_v = value
        i = 0
        for k, _ in self.records.items():
            self.records[k].append(value[i])
            i += 1

    def plot_sub(self):
        draw_sub(list(self.records.values()), list(self.records.keys()))

    def save(self, name):
        save_result(name, list(self.records.values()), list(self.records.keys()))
        print(f'result saved: {name}')

def get_last_result(file_names):
    for name in file_names:
        result, _ = read_result(name)
        acc = result[-1]
        acc = sum(acc[-5:]) / 5
        mse = result[-2]
        mse = sum(mse[-5:]) / 5
        print(name, 'acc: {:.4f}, mse: {:.4f}'.format(acc, mse))


def compare_save(file_names):
    loss = {}
    results = {}
    for name in file_names:
        result, columns = read_result(name)
        loss[name] = [result[0], result[1]]
        for i in range(2, len(columns)):
            if columns[i] not in results:
                results[columns[i]] = []
            results[columns[i]].append(result[i])
    '-- plot loss --'
    for k, v in loss.items():
        draw(v, ylabel='loss', ytags=['loss_g', 'loss_d'], title=k, save_name='loss_'+k)
    '-- plot other --'
    for k, v in results.items():
        draw(v, ylabel=k, title=k, ytags=file_names, save_name=k)



def sample_idx(n, count):
    """
    随机抽取序号
    :param n: 总数
    :param count: 抽取数量
    :return: 抽取的序号
    """
    arr = np.random.permutation(n)
    return arr[:count]


def save_result(file_name, result, columns_name):
    combined = list(map(list, zip(*result)))
    temp = pd.DataFrame(data=combined, columns=columns_name).reset_index(drop=True)
    temp.to_csv(os.path.join(result_path, file_name))


def build_logger(keys):
    logger = {}
    for i in keys:
        logger[i] = []
    return logger


def read_result(file_name):
    path = os.path.join(result_path, file_name)
    temp = pd.read_csv(path)
    columns = temp.columns
    columns = columns[1:]
    res = []
    for name in columns:
        res.append(temp[name].tolist())
    return res, columns


def _generate_mask(shape, rate_0):
    a = np.random.rand(*shape)
    mask = (a > rate_0) * 1
    return mask


def generate_mask_token(dim_x, dim_token, batch_size, rate_0):
    dim = dim_x // dim_token
    raw_mask = _generate_mask([batch_size, dim], rate_0)
    mask = np.zeros([batch_size, dim_x])
    st = 0
    for i in range(dim):
        for j in range(dim_token):
            mask[:, st + j] = raw_mask[:, i]
        st += dim_token
    return mask


def generate_mask_mix(info_list, batch_size, rate_0):
    dims = []
    for info in info_list:
        for i in info:
            if isinstance(i, SpanInfo):
                dims.append(i.dim)
    n = len(dims)
    raw_mask = _generate_mask([batch_size, n], rate_0)
    mask = np.zeros([batch_size, sum(dims)])
    st = 0
    for i in range(n):
        for j in range(dims[i]):
            mask[:, st + j] = raw_mask[:, i]
        st += dims[i]
    return mask


def generate_mask(train_x, test_x=None, rate_0=0.8, data_info=None, d_token=-1):
    if data_info is not None:
        train_m = generate_mask_mix(data_info, len(train_x), rate_0)
        if test_x is not None:
            test_m = generate_mask_mix(data_info, len(test_x), rate_0)
    elif d_token != -1:
        train_m = generate_mask_token(train_x.shape[-1], d_token, train_x.shape[0], rate_0)
        if test_x is not None:
            test_m = generate_mask_token(test_x.shape[-1], d_token, test_x.shape[0], rate_0)
    else:
        train_m = _generate_mask(train_x.shape, rate_0)
        if test_x is not None:
            test_m = _generate_mask(test_x.shape, rate_0)
    if test_x is not None:
        return train_m, test_m
    else:
        return train_m


def generate_noise(n, m):
    return np.random.uniform(0, 0.01, size=[n, m])
