import numpy as np
import torch
from numpy import argmax

from utils import utils


class Evaluator():
    def __init__(self, data, mask, output_info_list, device='cuda'):
        st, ed, missing_count = 0, 0, 0
        for column_info in output_info_list:
            if len(column_info) > 1:
                st += 1
                ed = st
            info = column_info[-1]
            ed += info.dim
            missing_count += np.count_nonzero(mask[:, st] == 0)
        self.missing_count = missing_count
        print(f'missing count:{self.missing_count}\n, '
              f'total feature count:{15 * len(data)}\n, '
              f'比例：{missing_count / (15 * len(data))}')

        data = torch.tensor(data, dtype=torch.float32, device=device)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        self.raw_data = data.cpu()
        self.data = data * mask
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

        imputed_x, _, _ = generator(input_g)
        loss_test = (torch.mean(((1 - self.mask) * imputed_x - (1 - self.mask) * self.data) ** 2)
                     / torch.mean(1 - self.mask))
        return loss_test

    def get_accuracy(self, generator, data_sampler, use_cond=False):
        if use_cond:
            cond, _, _, _ = data_sampler.sample_cond(len(self.data))
            cond = torch.tensor(cond, dtype=torch.float32, device='cuda')
            input_data = torch.cat(dim=1, tensors=[self.data, self.mask, cond])
        else:
            input_data = torch.concat(dim=1, tensors=[self.data, self.mask])
        imputed_data, _, _ = generator(input_data)
        imputed_data = self.data * self.mask + imputed_data * (1-self.mask)
        acc_count = 0
        start, end = 0, 0
        for column_info in self.output_info_list:
            if len(column_info) > 1:
                # skip continuous column
                start += 1
                end = start

            info = column_info[-1]
            end += info.dim
            v = imputed_data[:, start:end].argmax(axis=1).cpu()
            temp = self.raw_data[np.arange(len(self.data)), v+start] * (1 - self.mask[:, start]).cpu()
            acc_count += np.count_nonzero(temp == 1)
            start = end
        return acc_count/self.missing_count
