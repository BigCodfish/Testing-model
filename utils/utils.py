import os

import numpy as np
import pandas as pd

result_path = '../result/'
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), result_path)

def sample_idx(n, count):
    """
    随机抽取序号
    :param n: 总数
    :param count: 抽取数量
    :return: 抽取的序号
    """
    arr = np.random.permutation(n)
    return arr[:count]


def generate_mask(shape, p):
    a = np.random.rand(*shape)
    mask = (a > p) * 1
    return mask


def save_result(file_name, result, columns_name=['loss_g', 'loss_d', 'loss_test', 'w_distance', 'acc']):
    temp = pd.DataFrame(data=result, columns=columns_name).reset_index(drop=True)
    temp.to_csv(os.path.join(result_path, file_name))


def read_result(file_name):
    path = os.path.join(result_path, file_name)
    temp = pd.read_csv(path)
    columns = ['loss_g', 'loss_d', 'loss_test', 'w_distance', 'acc']
    res = []
    for name in columns:
        res.append(temp[name].tolist())
    return res

def generate_mask_token(dim_x, dim_token, batch_size, rate_0):
    dim = dim_x // dim_token
    raw_mask = generate_mask([batch_size, dim], rate_0)
    mask = np.zeros([batch_size, dim_x])
    st = 0
    for i in range(dim):
        for j in range(dim_token):
            mask[:, st+j] = raw_mask[:, i]
        st += dim_token
    return mask

def generate_mask_mix(info_list, batch_size, mask_rate):
    dims = []
    for info in info_list:
        for i in info:
            dims.append(i.dim)
    n = len(dims)
    raw_mask = generate_mask([batch_size, n], mask_rate)
    mask = np.zeros([batch_size, sum(dims)])
    st = 0
    for i in range(n):
        for j in range(dims[i]):
            mask[:, st + j] = raw_mask[:, i]
        st += dims[i]
    return mask


def generate_noise(n, m):
    return np.random.uniform(0, 0.01, size=[n, m])
