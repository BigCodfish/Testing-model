import os

import numpy as np
import torch
from torch import nn

from datautils.data_sampler import DataSampler
from datautils.data_transformer import preprocess, cross_validation, split_num_cat, load_ckpt
from model import train
from datautils import data_transformer as dt
from model.mlp import LayerConfig, MLP, MLP_SA
from model.vae.vae import Model_VAE, Encoder_model, Decoder_model
from model._vae import VanillaVAE
from trainer import trainer
from utils import painter, utils


def run_train_mlp_mlp_msn(use_cond):
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, data_info, dim_output = dt.transform(data, discrete_column)
    train_x, test_x = cross_validation(data, fixed=True, test_rate=0.2)
    if use_cond:
        data_sampler = DataSampler(data, data_info)
        dim_in = dim_output * 2 + data_sampler.n_categories
    else:
        dim_in = dim_output * 2
        data_sampler = None
    print(f'模型输入维数: {dim_in}')
    layers = [LayerConfig(dim_in, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'sigmoid'), ]
    generator = MLP(layers)
    discriminator = MLP(layers)
    logger = trainer.train(train_x, test_x, generator, discriminator, ['CTGAN', 'WD'],
                           data_info=data_info, data_sampler=data_sampler,
                           rate_0=0.2, hint_rate=0.9, batch_size=128, num_epochs=4000, device='cuda')
    return logger


def run_train_mlpsa_mlpsa_msn(use_cond, activate_all):
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, data_info, dim_output = dt.transform(data, discrete_column)
    train_x, test_x = cross_validation(data, fixed=True, test_rate=0.2)
    if use_cond:
        data_sampler = DataSampler(data, data_info)
        dim_in = dim_output * 2 + data_sampler.n_categories
    else:
        dim_in = dim_output * 2
        data_sampler = None
    print(f'模型输入维数: {dim_in}')
    layers = [LayerConfig(dim_in, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, ''), ]
    generator = MLP_SA(layers, data_info, activate_all=activate_all)
    discriminator = MLP_SA(layers, data_info, activate_all=activate_all)
    logger = trainer.train(train_x, test_x, generator, discriminator, ['GAIN', 'WD'],
                           data_info=data_info, data_sampler=data_sampler,
                           rate_0=0.2, hint_rate=0.9, batch_size=128, num_epochs=4000, device='cuda')
    return logger

def run_train_vae_mlp_msn(use_cond):
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, data_info, dim_output = dt.transform(data, discrete_column)
    train_x, test_x = cross_validation(data, fixed=True, test_rate=0.2)
    if use_cond:
        data_sampler = DataSampler(data, data_info)
        dim_in = dim_output * 2 + data_sampler.n_categories
    else:
        dim_in = dim_output * 2
        data_sampler = None
    print(f'模型输入维数: {dim_in}')
    layers = [LayerConfig(dim_in, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'sigmoid'), ]
    generator = VanillaVAE(dim_in, dim_output, 10, dim_output)
    discriminator = MLP(layers)
    logger = trainer.train(train_x, test_x, generator, discriminator, ['VAE', 'WD'],
                           data_info=data_info, data_sampler=data_sampler,
                           rate_0=0.2, hint_rate=0.9, batch_size=128, num_epochs=4000, device='cuda')
    return logger

def run_train_vae_mlpsa_msn(use_cond, activate_all):
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, data_info, dim_output = dt.transform(data, discrete_column)
    train_x, test_x = cross_validation(data, fixed=True, test_rate=0.2)
    if use_cond:
        data_sampler = DataSampler(data, data_info)
        dim_in = dim_output * 2 + data_sampler.n_categories
    else:
        dim_in = dim_output * 2
        data_sampler = None
    print(f'模型输入维数: {dim_in}')
    layers = [LayerConfig(dim_in, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, ''), ]
    generator = VanillaVAE(dim_in, dim_output, 10, dim_output)
    discriminator = MLP_SA(layers, data_info, activate_all=activate_all)
    logger = trainer.train(train_x, test_x, generator, discriminator, ['VAE', 'WD'],
                           data_info=data_info, data_sampler=data_sampler,
                           rate_0=0.2, hint_rate=0.9, batch_size=128, num_epochs=4000, device='cuda')
    return logger


def run_train_mlp_mlp_vae():
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    x_num, x_cat, dim_num, categories = preprocess(data, discrete_column)
    train_x_num, test_x_num, train_x_cat, test_x_cat = cross_validation(x_num, x_cat, fixed=True, test_rate=0.2)

    '-- preprocess --'
    train_x_num, test_x_num = torch.tensor(train_x_num).float().to('cuda'), torch.tensor(test_x_num).float().to('cuda')
    train_x_cat, test_x_cat = torch.tensor(train_x_cat).to('cuda'), torch.tensor(test_x_cat).to('cuda')
    pre_encoder = load_ckpt('adult', 'encoder.pt')
    pre_decoder = load_ckpt('adult', 'decoder.pt')
    train_x = pre_encoder(train_x_num, train_x_cat)
    test_x = pre_encoder(test_x_num, test_x_cat)
    train_x = train_x.view(train_x.shape[0], -1).detach().cpu().numpy()
    test_x = test_x.view(test_x.shape[0], -1).detach().cpu().numpy()

    dim_output = train_x.shape[-1]
    print(f'模型输入维数: {dim_output * 2}')
    layers = [LayerConfig(dim_output * 2, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'relu'),
              LayerConfig(dim_output, dim_output, 'sigmoid'), ]
    generator = MLP(layers)
    discriminator = MLP(layers)
    logger = trainer.train(train_x, test_x, generator, discriminator, ['GAIN', 'GAIN'],
                           d_token=4, decoder=pre_decoder,
                           rate_0=0.2, hint_rate=0.9, batch_size=128, num_epochs=4000, device='cuda')
    return logger

# logger = run_train_mlp_mlp_msn(True)
# logger = run_train_mlpsa_mlpsa_msn(True,True)
# logger = run_train_vae_mlpsa_msn(True, False)
logger = run_train_vae_mlp_msn(False)
# logger = run_train_mlp_mlp_vae()
logger.plot_sub()
logger.save('new_10')
