import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from evaluator import Evaluator
from model import loss
from datautils import data_transformer as dp
from model.layers import MixDataNet, MLP, build_net, build_trainer
from model.loss import CTGANLoss
from utils import utils
from utils.utils import generate_mask_mix

alpha = 10


def g_loss(x, new_x, mask, hint, cond, mask_cond, generator, discriminator, loss_type, output_info_list):
    input_g = torch.cat(dim=1, tensors=[new_x, mask])
    imputed_x = generator(input_g)
    hat_new_x = new_x * mask + imputed_x * (1 - mask)
    input_d = torch.cat(dim=1, tensors=[hat_new_x, hint])
    score = discriminator(input_d)

    loss_train = torch.mean((mask * new_x - mask * imputed_x) ** 2) / torch.mean(mask)
    loss_test = torch.mean(((1 - mask) * x - (1 - mask) * imputed_x) ** 2) / torch.mean(1 - mask)

    if loss_type == 'GAIN':
        loss_g = -torch.mean((1 - mask) * torch.log(score + 1e-8))
        loss_g = loss_g + alpha * loss_train
    elif loss_type == 'CTGAN':
        loss_fn = CTGANLoss(discriminator=discriminator, generator=generator, output_info_list=output_info_list)
        loss_g = loss_fn(data=new_x, mask=mask, hint=hint, cond=cond, mask_cond=mask_cond)
    else:
        raise ValueError(f'未知的generator损失函数类型{loss_type}')

    return loss_g, loss_train, loss_test


def d_loss(x, new_x, real_x, mask, hint, cond, mask_cond, generator, discriminator):
    input_g = torch.cat(dim=1, tensors=[new_x, mask])
    imputed_x = generator(input_g)
    input_d = torch.cat(dim=1, tensors=[imputed_x, hint])
    # discriminator的损失函数为wasserstein距离
    wasserstein_loss = loss.WDLoss(discriminator)
    grad_penalty, w_distance = wasserstein_loss(torch.cat(dim=1, tensors=[real_x, hint]), input_d)
    return grad_penalty, w_distance


def train(data, config_g, config_d, output_info_list, data_sampler, use_cond=False, batch_size=128, num_epochs=100,
          hint_rate=0.9):
    mask = generate_mask_mix(info_list=output_info_list, batch_size=len(data), mask_rate=0.2)
    train_data, test_data, train_mask, test_mask = dp.cross_validation(data, mask, True, 0.2)

    evaluator = Evaluator(data=test_data, mask=test_mask, output_info_list=output_info_list)

    real_data = train_data
    train_data *= train_mask
    test_data *= test_mask
    train_n, test_n = len(train_data), len(test_data)
    print(f"训练样本数量：{train_n}, 测试样本数量：{test_n}")
    dim = data.shape[-1]
    print(f"样本维数：{dim}")

    # 应用配置
    generator = build_net(config_g, output_info_list)
    discriminator = build_net(config_d, output_info_list)
    trainer_g = build_trainer(config_g, generator)
    trainer_d = build_trainer(config_d, discriminator)

    loss_g_list = []
    loss_d_list = []
    w_distances = []
    loss_test = []
    acc_list = []
    iterator = tqdm(range(num_epochs))
    for epoch in iterator:
        # 采样训练数据
        idx = utils.sample_idx(train_n, batch_size)
        x = train_data[idx, :]
        m = train_mask[idx, :]
        real_x = train_data[idx, :]
        # 噪声生成方式修改？
        noise = utils.generate_noise(batch_size, dim)
        new_x = x * m + (1 - m) * noise
        hint = generate_mask_mix(info_list=output_info_list, batch_size=batch_size, mask_rate=1-hint_rate)
        hint *= m
        # mask_1 = np.count_nonzero(m == 1)
        # hint_1 = np.count_nonzero(hint == 1)
        # size = m.shape[0] * m.shape[1]
        # print(f'mask:{mask_1/size}, hint:{hint_1/size}')
        # 获取条件矩阵
        c, m_c, i, j = data_sampler.sample_cond(batch_size)

        m_c = torch.tensor(m_c, dtype=torch.float32, device='cuda')
        c = torch.tensor(c, dtype=torch.float32, device='cuda')
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
        m = torch.tensor(m, dtype=torch.float32, device='cuda')
        hint = torch.tensor(hint, dtype=torch.float32, device='cuda')
        new_x = torch.tensor(new_x, dtype=torch.float32, device='cuda')
        real_x = torch.tensor(real_x, dtype=torch.float32, device='cuda')

        trainer_d.zero_grad()
        loss_d, w_distance = d_loss(x=x, real_x=real_x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c,
                                    generator=generator, discriminator=discriminator)
        loss_d.backward()
        trainer_d.step()

        trainer_g.zero_grad()
        loss_g, MSE_train, MSE_test = g_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c,
                                             loss_type=config_g.loss, output_info_list=output_info_list,
                                             generator=generator, discriminator=discriminator)
        loss_g.backward()
        trainer_g.step()

        if epoch % 1 == 0:
            loss_g_list.append(loss_g.item())
            loss_d_list.append(loss_d.item())
            w_distances.append(w_distance.item())
            t = evaluator.test(data_sampler, generator, use_cond=use_cond)
            acc = evaluator.get_accuracy(generator=generator)
            acc_list.append(acc)
            loss_test.append(t.item())

    return loss_g_list, loss_d_list, w_distances, loss_test, acc_list
