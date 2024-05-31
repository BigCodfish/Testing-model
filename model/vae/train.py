import os
import sys

import torch
from torch import nn
from tqdm import tqdm

import datautils.data_transformer as dt
from model.vae.model_vae import VAE
from utils import utils



def _compute_loss(x_num, x_cat, re_x_num, re_x_cat, mu, var):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (x_num - re_x_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x in enumerate(re_x_cat):
        if x is not None:
            ce_loss += ce_loss_fn(x, x_cat[:, idx])
            x_hat = x.argmax(dim=-1)
        acc += (x_hat == x_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= (idx + 1)
    acc /= total_num

    temp = 1 + var - mu.pow(2) - var.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_VAE():
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    x_num, x_cat, dim_num, categories = dt.preprocess(data, discrete_column)
    train_x_num, test_x_num = dt.cross_validation(x_num, test_rate=0.3)
    train_x_cat, test_x_cat = dt.cross_validation(x_cat, test_rate=0.3)

    num_epoch = 50000
    batch_size = 4096
    dim_token = 4
    in_dim = (dim_num + len(categories)+1) * dim_token
    train_n = len(train_x_cat)

    test_x_num = torch.tensor(test_x_num, dtype=torch.float32, device='cuda')
    test_x_cat = torch.tensor(test_x_cat, dtype=torch.int64, device='cuda')

    model = VAE(dim_num, categories, dim_token, in_dim, hidden_dim=in_dim//2, latent_dim=10)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e3, weight_decay=0)
    iterator = tqdm(range(num_epoch))

    for epoch in iterator:
        idx = utils.sample_idx(train_n, batch_size)
        batch_num = train_x_num[idx, :]
        batch_cat = train_x_cat[idx, :]

        batch_num = torch.tensor(batch_num, dtype=torch.float32, device='cuda')
        batch_cat = torch.tensor(batch_cat, dtype=torch.int64, device='cuda')

        optimizer.zero_grad()
        re_x_num, re_x_cat, mu, var = model(batch_num, batch_cat)

        loss_mse, loss_ce, loss_kl, train_acc = _compute_loss(batch_num, batch_cat, re_x_num, re_x_cat, mu, var)
        loss = loss_mse + loss_ce + loss_kl
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                re_x_num, re_x_cat, mu, var = model(test_x_num, test_x_cat)
                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = _compute_loss(test_x_num, test_x_cat,
                                                                                re_x_num, re_x_cat, mu, var)
                print(
                    'epoch: {},Train MSE:{:6f}, Train CE:{:6f}, Train ACC:{:6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Val ACC:{:6f}'.format(
                        epoch, loss_mse.item(), loss_ce.item(),train_acc.item(),
                        val_mse_loss.item(), val_ce_loss.item(), val_acc.item()))
            model.train()

train_VAE()