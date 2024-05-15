import numpy as np
import torch

from datautils.data_transformer import SpanInfo
from utils.utils import generate_mask_mix


def _test_data_mask():
    data = np.ones([10, 10])
    info_list = [[SpanInfo(dim=2, activation_fn=''), SpanInfo(dim=1, activation_fn='')],
                 [SpanInfo(dim=3, activation_fn=''), SpanInfo(dim=4, activation_fn='')]]
    mask = generate_mask_mix(info_list=info_list, batch_size=len(data), mask_rate=0.5)
    masked = data * mask
    print(data)
    print(masked)
    t = data-masked
    print(np.count_nonzero(t == 0) / 100)

def _test_evaluator():
    data = np.ones([10, 10])
    info_list = [[SpanInfo(dim=2, activation_fn=''), SpanInfo(dim=1, activation_fn='')],
                 [SpanInfo(dim=3, activation_fn=''), SpanInfo(dim=4, activation_fn='')]]
    n = len(data)
    mask = generate_mask_mix(info_list=info_list, batch_size=len(data), mask_rate=0.5)
    masked = data * mask
    missing_count = 0
    acc_count = 0
    for i in range(n):
        start, end = 0, 0
        for column_info in info_list:
            if len(column_info) > 1:
                # 跳过一个连续变量
                start += 1
                end = start

            info = column_info[-1]
            end += info.dim
            if mask[i, start] == 0:
                missing_count += 1

            v = data[i, start:end].argmax(axis=1)
            if data[i, v] == 1:
                acc_count += 1
            start = end
    print(f'离散数据缺失数：{missing_count}'
          f'离散数据预测准确率：{acc_count / missing_count}')

# _test_data_mask()
data = np.array([[1,2,3],
                [3,5,4]])
print(data.argmax(axis=1))
# _test_evaluator()