import numpy as np

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


def generate_noise(n,m):
    return np.random.uniform(0, 0.01, size=[n, m])