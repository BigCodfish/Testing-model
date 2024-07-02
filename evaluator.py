import numpy as np
import pandas as pd
import torch
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from datautils import data_transformer as dt
from model._vae import VanillaVAE
from utils import utils


class Evaluator():
    def __init__(self, test_x, test_m, rate_0, dim_token=-1, decoder=None, output_info_list=None, device='cuda'):
        self.rate_0 = rate_0
        self.device = device
        self.x = test_x
        self.m = test_m
        self.output_info_list = output_info_list
        self.dim_token = dim_token
        self.decoder = decoder

    def eval_msn(self, generator, data_sampler=None):
        noise = utils.generate_noise(self.x.shape[0], self.x.shape[1])
        new_x = self.m * self.x + (1 - self.m) * noise

        m = torch.tensor(self.m, dtype=torch.float32, device=self.device)
        x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=self.device)

        if data_sampler is not None:
            c, _, _, _ = data_sampler.sample_cond(len(x))
            c = torch.tensor(c, dtype=torch.float32, device=self.device)
            input_g = torch.cat(dim=1, tensors=[new_x, m, c])
        else:
            input_g = torch.cat(dim=1, tensors=[new_x, m])

        if isinstance(generator, VanillaVAE):
            imputed_x, _, _ = generator(input_g)
        else:
            imputed_x = generator(input_g)
        imputed_x = imputed_x * (1-m) + x * m
        acc = self.compute_acc_msn(imputed_x, x)
        mse = self.compute_mse_msn(imputed_x, x)
        return mse, acc

    def compute_mse_msn(self, imputed_x, x):
        st, ed = 0, 0
        total = 0
        mse = 0
        for column_info in self.output_info_list:
            if len(column_info) == 1:
                st += column_info[0].dim
                ed = st
                continue
            transformer = column_info[-1]
            column_name = column_info[-2]
            ed += (1 + column_info[1].dim)
            i_x_num = imputed_x[:, st:ed].detach().cpu().numpy()
            x_num = x[:, st:ed].detach().cpu().numpy()
            i_x_num = dt.reverse_transform_num(i_x_num, column_name, transformer)
            x_num = dt.reverse_transform_num(x_num, column_name, transformer)
            ' 重新max-min标准化， 方便计算mse损失 '
            norm = MinMaxScaler()
            norm.fit(x_num, column_name)
            x_num = norm.transform(x_num)
            i_x_num = norm.transform(i_x_num)
            i_x_num = torch.tensor(i_x_num, dtype=torch.float32, device=self.device)
            x_num = torch.tensor(x_num, dtype=torch.float32, device=self.device)
            mse += nn.functional.mse_loss(x_num, i_x_num, reduction='sum')
            total += x.shape[0]
        mse = mse / (total * self.rate_0)
        return mse

    def compute_acc_msn(self, imputed_x, x):
        start, end = 0, 0
        acc, total = 0, 0
        for column_info in self.output_info_list:
            if len(column_info) > 1:
                # skip continuous column
                info = column_info[1]
                start += (1 + info.dim)
                end = start
                continue
            info = column_info[0]
            end += info.dim
            v = imputed_x[:, start:end].argmax(dim=1)
            temp = x[np.arange(len(x)), v + start]
            temp = temp.cpu().numpy()
            acc += np.count_nonzero(temp == 1)
            total += x.shape[0]
            start = end
        if total == 0:
            return 1
        else:
            acc /= total
            acc = (acc - (1 - self.rate_0)) / self.rate_0
            return acc

    def eval_vae(self, generator):

        z = utils.generate_noise(self.x.shape[0], self.x.shape[1])
        new_x = self.x * self.m + z * (1 - self.m)

        x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        m = torch.tensor(self.m, dtype=torch.float32, device=self.device)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=self.device)
        test_x = torch.tensor(self.x, dtype=torch.float32, device=self.device)

        input_g = torch.cat(dim=1, tensors=[new_x, m])
        impute_x = generator(input_g)
        impute_x = x * m + impute_x * (1 - m)
        impute_x = impute_x.view(test_x.shape[0], -1, self.dim_token)
        imputed_x_num, impute_x_cat = self.decoder(impute_x[:, 1:])
        test_x = test_x.view(test_x.shape[0], -1, self.dim_token)
        test_x_num, test_x_cat = self.decoder(test_x[:, 1:])

        miss_count = test_x_num.shape[0] * test_x_num.shape[1] * self.rate_0
        mse = nn.functional.mse_loss(imputed_x_num, test_x_num, reduction='sum') / miss_count
        acc = self.compute_acc_vae(impute_x_cat, test_x_cat)
        return mse, acc

    def compute_acc_vae(self, imputed_x, x):
        acc, total_num = 0, 0
        for idx, x_cat in enumerate(imputed_x):
            if x_cat is not None:
                x_hat = x_cat.argmax(dim=-1)
                test_x_hat = x[idx].argmax(dim=-1)
            acc += (x_hat == test_x_hat).float().sum()
            total_num += x_hat.shape[0]
        acc /= total_num
        acc = (acc - (1 - self.rate_0)) / self.rate_0
        return acc.item()