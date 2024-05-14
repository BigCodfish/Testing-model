import torch
from torch import nn
from tqdm import tqdm

from model import loss
from datautils import data_transformer as dp
from model.layers import MixDataNet, MLP
from model.loss import CTGANLoss
from utils import utils

alpha = 10


def g_loss(x, new_x, mask, hint, cond, mask_cond, generator, discriminator, loss_type, output_info_list):
    """
        :param x: 原数据；
        :param new_x：加噪后数据；
        :param mask：掩码；
        :param hint:提示数据
    """
    input_g = torch.cat(dim=1, tensors=[new_x, mask, cond])
    imputed_x = generator(input_g)
    hat_new_x = new_x * mask + imputed_x * (1 - mask)
    input_d = torch.cat(dim=1, tensors=[hat_new_x, hint, cond])
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


def d_loss(x, new_x, mask, hint, cond, mask_cond, generator, discriminator):
    input_g = torch.cat(dim=1, tensors=[new_x, mask, cond])
    imputed_x = generator(input_g)
    input_d = torch.cat(dim=1, tensors=[imputed_x, hint, cond])
    # discriminator的损失函数为wasserstein距离
    wasserstein_loss = loss.WDLoss(discriminator)
    grad_penalty, w_distance = wasserstein_loss(torch.cat(dim=1, tensors=[x, hint, cond]), input_d)
    return grad_penalty, w_distance


def _test_loss(x, new_x, mask, cond, generator):
    input_g = torch.cat(dim=1, tensors=[new_x, mask, cond])
    imputed_x = generator(input_g)
    loss_test = torch.mean(((1-mask)*imputed_x - (1-mask)*x)**2) / torch.mean(1-mask)
    return loss_test, imputed_x


def test(x, mask, data_sampler, generator):
    noise = utils.generate_noise(x.shape[0], x.shape[1])
    new_x = mask * x + (1 - mask) * noise
    cond, _, _, _ = data_sampler.sample_cond(len(x))

    cond = torch.tensor(cond, dtype=torch.float32, device='cuda')
    x = torch.tensor(x, dtype=torch.float32, device='cuda')
    new_x = torch.tensor(new_x, dtype=torch.float32, device='cuda')
    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')

    loss_t, imputed_x = _test_loss(x=x, new_x=new_x, mask=mask, cond=cond, generator=generator)

    return loss_t


def build_net(config, output_info_list, device='cuda'):
    if config.type == 'MixDataNet':
        return MixDataNet(config=config, output_info_list=output_info_list).to(device)
    elif config.type == 'MLP':
        return MLP(config=config, output_info_list=output_info_list).to(device)
    else:
        raise ValueError(f'未知的网络类型：{config.type}')


def build_trainer(config, net):
    if config.optim == 'Adam':
        return torch.optim.Adam(net.parameters())
    else:
        raise ValueError(f'未知的优化器类型{config.optim}')


def train(data, config_g, config_d, output_info_list, data_sampler, batch_size=128, num_epochs=100, hint_rate=0.9):
    mask = utils.generate_mask(data.shape, 0.2)
    train_data, test_data, train_mask, test_mask = dp.cross_validation(data, mask, True, 0.2)
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
    iterator = tqdm(range(num_epochs))
    for epoch in iterator:
        # 采样训练数据
        idx = utils.sample_idx(train_n, batch_size)
        x = train_data[idx, :]
        m = train_mask[idx, :]
        # 噪声生成方式修改？
        noise = utils.generate_noise(batch_size, dim)
        new_x = x * m + (1 - m) * noise
        hint = utils.generate_mask([batch_size, dim], 1 - hint_rate)
        hint *= m
        # 获取条件矩阵
        c, m_c, i, j = data_sampler.sample_cond(batch_size)

        m_c = torch.tensor(m_c, dtype=torch.float32, device='cuda')
        c = torch.tensor(c, dtype=torch.float32, device='cuda')
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
        m = torch.tensor(m, dtype=torch.float32, device='cuda')
        hint = torch.tensor(hint, dtype=torch.float32, device='cuda')
        new_x = torch.tensor(new_x, dtype=torch.float32, device='cuda')

        trainer_d.zero_grad()
        loss_d, w_distance = d_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c,
                                              generator=generator, discriminator=discriminator)
        loss_d.backward()
        trainer_d.step()

        trainer_g.zero_grad()
        loss_g, MSE_train, MSE_test = g_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c, loss_type=config_g.loss,
                                             output_info_list=output_info_list,
                                             generator=generator, discriminator=discriminator)
        loss_g.backward()
        trainer_g.step()

        if epoch % 1 == 0:
            loss_g_list.append(loss_g.item())
            loss_d_list.append(loss_d.item())
            w_distances.append(w_distance.item())
            t = test(test_data, test_mask, data_sampler, generator)
            loss_test.append(t.item())

    return loss_g_list, loss_d_list, w_distances, loss_test
