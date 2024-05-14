import torch
from numpy import argmax


def test_accuracy(data, mask, model, output_info_list):
    data = torch.tensor(data, dtype=torch.float32, device='cuda')
    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
    input_data = torch.concat(dim=1, tensors=[data, mask])
    imputed_data = model(input_data)
    imputed_data = data * mask + imputed_data * (1-mask)
    n = len(data)
    missing_count = 0
    acc_count = 0
    for i in range(n):
        start, end = 0, 0
        for column_info in output_info_list:
            if len(column_info) > 1:
                start += (column_info[1].dim + 1)
                end = start
                continue
            for info in column_info:
                end += info.dim
                if mask[i, start] == 0:
                    missing_count += 1
                    v, t = torch.max(imputed_data[i, start:end])
                    if v == 1:
                        acc_count += 1
                start = end
    print(f'离散数据缺失数：{missing_count}'
          f'离散数据预测准确率：{acc_count/missing_count}')



