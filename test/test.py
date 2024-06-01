import numpy as np
import torch
from torch import nn

from datautils.data_transformer import SpanInfo
from utils.utils import generate_mask_mix


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    return mse_loss, ce_loss, acc

def test_model_save(pre_decoder, model_save_path, encoder_save_path, decoder_save_path, x_num_test, x_cat_test, re_x_num, re_x_cat, mu, var, x, h):
    print('testing model save and reload...')
    model = torch.load(model_save_path)
    encoder = torch.load(encoder_save_path)
    decoder = pre_decoder
    encode_mu = encoder(x_num_test, x_cat_test)
    encode_re_x_num, encode_re_x_cat, encode_h = decoder(encode_mu[:, 1:])
    hat_re_x_num, hat_re_x_cat, hat_mu, hat_var, hat_x, hat_h = model(x_num_test, x_cat_test)

    print(f're_x_num-encode:{encode_re_x_num.equal(re_x_num)}, re_x_num-hat:{re_x_num.equal(hat_re_x_num)}')
    for i in range(len(re_x_cat)):
        if re_x_cat[i].equal(encode_re_x_cat[i]) is False:
            print(f'x_cat not equal to encode_re_x_cat')
            break
        if re_x_cat[i].equal(hat_re_x_cat[i]) is False:
            print(f'x_cat not equal to hat_re_x_cat')
            break

    print(f'mu-encode:{encode_mu.equal(mu)}, mu-hat:{hat_mu.equal(mu)}')
    print(f'var:{hat_var.equal(var)}')
    print(f'x:{hat_x.equal(x)}')
    print(f'h-encode:{encode_h.equal(h)}, h-hat:{hat_h.equal(h)}')
    mse, ce, acc = compute_loss(x_num_test, x_cat_test, encode_re_x_num, encode_re_x_cat)
    print(f'encode - mse:{mse}, ce:{mse}, acc:{acc}')
    mse, ce, acc = compute_loss(x_num_test, x_cat_test, hat_re_x_num, hat_re_x_cat)
    print(f'hat - mse:{mse}, ce:{mse}, acc:{acc}')


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
# data = np.array([[1,2,3],
#                 [3,5,4]])
# print(data.argmax(axis=1))
# _test_evaluator()