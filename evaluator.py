import torch
from numpy import argmax

from utils import utils


class Evaluator():
    def __init__(self, data, mask, output_info_list, device='cuda'):
        data = torch.tensor(data, dtype=torch.float32, device=device)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        self.data = data
        self.mask = mask
        self.output_info_list = output_info_list

    def test(self, data_sampler, generator, use_cond=False):
        noise = utils.generate_noise(self.data.shape[0], self.data.shape[1])
        noise = torch.tensor(noise, dtype=torch.float32, device='cuda')
        new_x = self.mask * self.data + (1 - self.mask) * noise
        if use_cond:
            cond, _, _, _ = data_sampler.sample_cond(len(self.data))
            cond = torch.tensor(cond, dtype=torch.float32, device='cuda')
            input_g = torch.cat(dim=1, tensors=[new_x, self.mask, cond])
        else:
            input_g = torch.cat(dim=1, tensors=[new_x, self.mask])

        imputed_x = generator(input_g)
        loss_test = (torch.mean(((1 - self.mask) * imputed_x - (1 - self.mask) * self.data) ** 2)
                     / torch.mean(1 - self.mask))
        return loss_test

    def get_accuracy(self, generator):
        input_data = torch.concat(dim=1, tensors=[self.data, self.mask])
        imputed_data = generator(input_data)
        imputed_data = self.data * self.mask + imputed_data * (1-self.mask)
        n = len(self.data)
        missing_count = 0
        acc_count = 0
        for i in range(n):
            start, end = 0, 0
            for column_info in self.output_info_list:
                if len(column_info) > 1:
                    start += (column_info[1].dim + 1)
                    end = start
                    continue
                for info in column_info:
                    end += info.dim
                    if self.mask[i, start] == 0:
                        missing_count += 1
                        v, t = torch.max(imputed_data[i, start:end])
                        if v == 1:
                            acc_count += 1
                    start = end
        print(f'离散数据缺失数：{missing_count}'
              f'离散数据预测准确率：{acc_count/missing_count}')



