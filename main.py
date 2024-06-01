import os

import numpy as np
import torch
from torch import nn

from datautils.data_sampler import DataSampler
from datautils.data_transformer import preprocess, cross_validation, split_num_cat
from model import train
from datautils import data_transformer as dt
from model.layers import NetConfig, LayerConfig
from model.train import train_token
from model.vae.vae import Model_VAE, Encoder_model, Decoder_model
from utils import painter, utils

def run_train():
    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    data, output_info_list, output_dim = dt.transform(data, discrete_column)
    # data = dt.mask_data(raw_data=data, info_list=output_info_list, mask_rate=0.7)
    data_sampler = DataSampler(data, output_info_list)
    cond_dim = data_sampler.n_categories
    print(f"模型输入维数：{output_dim * 2}")
    layers = [LayerConfig(output_dim * 2, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'sigmoid')]
    layers_VAE = [LayerConfig(output_dim * 2, output_dim, '', ''),
                  LayerConfig(10, output_dim, '', '')]

    # VAE
    config_g = NetConfig(type='VAE', layers=layers_VAE, optim='Adam', loss='VAE')
    config_d = NetConfig(type='MLP', layers=layers, optim='Adam', loss='WD')

    # 逐层分段激活
    # config_g = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='CTGAN')
    # config_d = NetConfig(type='MixDataNet', layers=layers, optim='Adam', loss='WD')

    # 不逐层分段激活
    # config_g = NetConfig(type='MLP', layers=layers, optim='Adam', loss='CTGAN')
    # config_d = NetConfig(type='MLP', layers=layers, optim='Adam', loss='WD')

    loss_g, loss_d, w_distance, loss_test, acc = train.train(data, config_g, config_d, output_info_list, data_sampler,
                                                     batch_size=128, num_epochs=500, use_cond=False)
    combined = list(map(list, zip(loss_g, loss_d, loss_test, w_distance, acc)))
    utils.save_result('10', combined)
    painter.draw_sub([loss_g, loss_d, w_distance, loss_test, acc])

def run_train_2():
    device = 'cuda'
    data_name = 'adult'
    num_layers = 2
    d_token = 4
    token_bias = True
    n_head = 1
    factor = 32

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/model/vae/ckpt/{data_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    data, discrete_column = dt.read_csv('adult.csv', 'adult.json')
    X_num, X_cat, d_numerical, categories = preprocess(data, discrete_column)

    model = torch.load(model_save_path)
    model = model.to(device)

    pre_encoder = Encoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)
    pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)
    pre_encoder.load_weights(model)
    pre_decoder.load_weights(model)

    X_num = torch.tensor(X_num).float().to(device)
    X_cat = torch.tensor(X_cat).to(device)

    z = pre_encoder(X_num, X_cat)

    x = z.view(z.shape[0], -1)
    output_dim = x.shape[-1]
    layers = [LayerConfig(output_dim * 2, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'relu'),
              LayerConfig(output_dim, output_dim, 'Linear', 'sigmoid')]
    config_g = NetConfig(type='MLP', layers=layers, optim='Adam', loss='GAIN')
    config_d = NetConfig(type='MLP', layers=layers, optim='Adam', loss='GAIN')

    l_g, l_d, wd, mse, acc = train_token(x, d_token, config_g, config_d, decoder=pre_decoder,num_epochs=5000)
    painter.draw_sub([l_g, l_d, wd, mse, acc])


run_train_2()

def comparison_1_2():
    res_1 = utils.read_result('1')
    res_2 = utils.read_result('2')
    painter.draw([res_1[0], res_1[1]], ylabel='loss', ytags=['loss_g', 'loss_d'],title='loss in training 1')
    painter.draw([res_2[0], res_2[1]], ylabel='loss', ytags=['loss_g', 'loss_d'], title='loss in training 2')
    painter.draw([res_1[2], res_2[2]], ylabel='w_distance', ytags=['GAIN loss', 'WDist loss'], title='w_distance')
    painter.draw([res_1[3], res_2[3]], ylabel='loss test', ytags=['GAIN loss', 'WDist loss'], title='loss test')
    painter.draw([res_1[4], res_2[4]], ylabel='test acc', ytags=['GAIN loss', 'WDist loss'], title='test acc')
