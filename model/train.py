import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from evaluator import Evaluator
from model import loss
from datautils import data_transformer as dp
from model.layers import MixDataNet, MLP, build_net, build_trainer
from model.loss import CTGANLoss, InformationLoss
from utils import utils
from utils.utils import generate_mask_mix, generate_mask_token

alpha = 10


def g_loss(x, new_x, mask, hint, cond, mask_cond, generator, discriminator, loss_type, output_info_list, use_cond):
    if use_cond:
        input_g = torch.cat(dim=1, tensors=[new_x, mask, cond])
        if loss_type == 'VAE':
            imputed_x, mu, var = generator(input_g)
        else:
            imputed_x = generator(input_g)
        imputed_x = new_x * mask + imputed_x * (1 - mask)
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint, cond])
    else:
        input_g = torch.cat(dim=1, tensors=[new_x, mask])
        if loss_type == 'VAE':
            imputed_x, mu, var = generator(input_g)
        else:
            imputed_x = generator(input_g)
        imputed_x = new_x * mask + imputed_x * (1 - mask)
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint])

    score = discriminator(input_d)

    if loss_type == 'GAIN':
        loss_g = -torch.mean((1 - mask) * torch.log(score + 1e-8))
        loss_train = torch.mean((mask * new_x - mask * imputed_x) ** 2) / torch.mean(mask)
        loss_g = loss_g + alpha * loss_train
    elif loss_type == 'CTGAN':
        loss_fn = CTGANLoss(discriminator=discriminator, generator=generator, output_info_list=output_info_list)
        loss_g = loss_fn(data=new_x, mask=mask, hint=hint, cond=cond, mask_cond=mask_cond)
    elif loss_type == 'Information':
        loss_fn = InformationLoss(generator=generator, discriminator=discriminator)
        loss_g = loss_fn(data=new_x, mask=mask, hint=hint, cond=cond)
    elif loss_type == 'VAE':
        BCE = nn.functional.cross_entropy(imputed_x, x, reduction='mean')
        KLD = torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))
        loss_g = BCE + KLD
        loss_g *= 0.1
        loss_g += torch.mean((1 - mask) * torch.log(score + 1e-8))
    else:
        raise ValueError(f'未知的generator损失函数类型{loss_type}')

    return loss_g


def d_loss(x, new_x, mask, hint, cond, mask_cond, loss_type, generator, discriminator, use_cond):
    if use_cond:
        input_g = torch.cat(dim=1, tensors=[new_x, mask, cond])
        imputed_x, _, _ = generator(input_g)
        hat_new_x = new_x * mask + imputed_x * (1 - mask)
        input_d = torch.cat(dim=1, tensors=[hat_new_x, hint, cond])
        input_d_real = torch.cat(dim=1, tensors=[x, mask, cond])
    else:
        input_g = torch.cat(dim=1, tensors=[new_x, mask])
        imputed_x = generator(input_g)
        hat_new_x = new_x * mask + imputed_x * (1 - mask)
        input_d = torch.cat(dim=1, tensors=[hat_new_x, hint])
        input_d_real = torch.cat(dim=1, tensors=[x, mask])

    # discriminator的损失函数为wasserstein距离
    wasserstein_loss = loss.WDLoss(discriminator)
    loss_d, w_distance = wasserstein_loss(input_d_real, input_d)
    if loss_type == 'GAIN':
        estimate = discriminator(input_d)
        loss_d = -torch.mean(mask * torch.log(estimate + 1e-8) + (1 - mask) * torch.log(1. - estimate + 1e-8))
    return loss_d, w_distance


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
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

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def evaluate(test_x, test_m, generator, dim_token, device='cuda', decoder=None):
    with torch.no_grad():
        x = test_x * test_m
        z = utils.generate_noise(test_x.shape[0], test_x.shape[1])
        new_x = x * test_m + z * (1 - test_m)

        x = torch.tensor(x, dtype=torch.float32, device=device)
        m = torch.tensor(test_m, dtype=torch.float32, device=device)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=device)
        test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

        input_g = torch.cat(dim=1, tensors=[new_x, m])
        impute_x = generator(input_g)
        impute_x = x * m + impute_x * (1 - m)
        impute_x = impute_x.view(test_x.shape[0], -1, dim_token)
        imputed_x_num, impute_x_cat = decoder(impute_x[:, 1:])
        test_x = test_x.view(test_x.shape[0], -1, dim_token)
        test_x_num, test_x_cat = decoder(test_x[:, 1:])

        mse = ((test_x_num - imputed_x_num) ** 2).mean()
        acc, total_num = 0, 0
        for idx, x_cat in enumerate(impute_x_cat):
            if x_cat is not None:
                x_hat = x_cat.argmax(dim=-1)
                test_x_hat = test_x_cat[idx].argmax(dim=-1)
            acc += (x_hat == test_x_hat).float().sum()
            total_num += x_hat.shape[0]
        acc /= total_num

    return mse, acc


def train_token(x, dim_token, config_g, config_d, batch_size=128, num_epochs=5000, device='cuda', hint_rate=0.9,
                decoder=None):
    m = generate_mask_token(x.shape[-1], dim_token, x.shape[0], rate_0=0.2)
    train_x, test_x, train_m, test_m = dp.cross_validation(x, m, test_rate=0.2)
    train_x, test_x = train_x.cpu().detach().numpy(), test_x.cpu().detach().numpy()
    train_x *= train_m
    train_n, test_n, dim = len(train_x), len(test_x), train_x.shape[-1]
    print(f'训练数据维度：{dim}, 训练数据量：{train_n}， 测试数据量：{test_n}')

    generator = build_net(config_g).to(device)
    discriminator = build_net(config_d).to(device)
    trainer_g = build_trainer(config_g, generator)
    trainer_d = build_trainer(config_d, discriminator)

    loss_g_list = []
    loss_d_list = []
    wd_list = []
    mse_list = []
    acc_list = []

    iterator = tqdm(range(num_epochs))
    for epoch in iterator:
        idx = utils.sample_idx(train_n, batch_size)
        x = train_x[idx, :]
        m = train_m[idx, :]
        z = utils.generate_noise(batch_size, dim)
        new_x = x * m + z * (1 - m)
        hint = generate_mask_token(dim, dim_token, batch_size, 1 - hint_rate)
        hint *= m

        x = torch.tensor(x, dtype=torch.float32, device=device)
        m = torch.tensor(m, dtype=torch.float32, device=device)
        hint = torch.tensor(hint, dtype=torch.float32, device=device)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=device)

        trainer_d.zero_grad()
        loss_d, w_distance = d_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=None, mask_cond=False,
                                    loss_type=config_d.loss,
                                    generator=generator, discriminator=discriminator, use_cond=False)
        loss_d.backward()
        trainer_d.step()

        trainer_g.zero_grad()
        loss_g = g_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=None, mask_cond=None,
                        loss_type=config_g.loss, output_info_list=None,
                        generator=generator, discriminator=discriminator, use_cond=False)
        loss_g.backward()
        trainer_g.step()
        '--Evaluation--'
        if epoch % 1 == 0:
            generator.eval()
            discriminator.eval()
            loss_g_list.append(loss_g.item())
            loss_d_list.append(loss_d.item())
            wd_list.append(w_distance.item())
            mse, acc = evaluate(test_x, test_m, generator, dim_token, decoder=decoder)
            acc_list.append(acc.item())
            mse_list.append(mse.item())

            generator.train()
            discriminator.train()

    return loss_g_list, loss_d_list, wd_list, mse_list, acc_list


def train(data, config_g, config_d, output_info_list, data_sampler, use_cond=False, batch_size=128, num_epochs=100,
          hint_rate=0.9):
    mask = generate_mask_mix(info_list=output_info_list, batch_size=len(data), mask_rate=0.2)
    train_data, test_data, train_mask, test_mask = dp.cross_validation(data, mask, True, 0.2)

    evaluator = Evaluator(data=test_data, mask=test_mask, output_info_list=output_info_list)

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
        # 噪声生成方式修改？
        noise = utils.generate_noise(batch_size, dim)
        new_x = x * m + (1 - m) * noise
        hint = generate_mask_mix(info_list=output_info_list, batch_size=batch_size, mask_rate=1 - hint_rate)
        hint *= m
        # mask_1 = np.count_nonzero(m == 1)
        # hint_1 = np.count_nonzero(hint == 1)
        # size = m.shape[0] * m.shape[1]
        # print(f'mask:{mask_1/size}, hint:{hint_1/size}')
        # 获取条件矩阵
        if use_cond:
            c, m_c, i, j = data_sampler.sample_cond(batch_size)
            c = torch.tensor(c, dtype=torch.float32, device='cuda')
            m_c = torch.tensor(m_c, dtype=torch.float32, device='cuda')
        else:
            c, m_c, i, j = None, None, None, None

        x = torch.tensor(x, dtype=torch.float32, device='cuda')
        m = torch.tensor(m, dtype=torch.float32, device='cuda')
        hint = torch.tensor(hint, dtype=torch.float32, device='cuda')
        new_x = torch.tensor(new_x, dtype=torch.float32, device='cuda')

        trainer_d.zero_grad()
        loss_d, w_distance = d_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c,
                                    loss_type=config_d.loss,
                                    generator=generator, discriminator=discriminator, use_cond=use_cond)
        loss_d.backward()
        trainer_d.step()

        trainer_g.zero_grad()
        loss_g = g_loss(x=x, new_x=new_x, mask=m, hint=hint, cond=c, mask_cond=m_c,
                        loss_type=config_g.loss, output_info_list=output_info_list,
                        generator=generator, discriminator=discriminator, use_cond=use_cond)
        loss_g.backward()
        trainer_g.step()

        if epoch % 1 == 0:
            loss_g_list.append(loss_g.item())
            loss_d_list.append(loss_d.item())
            w_distances.append(w_distance.item())
            t = evaluator.test(data_sampler, generator, use_cond=use_cond)
            acc = evaluator.get_accuracy(generator=generator, data_sampler=data_sampler, use_cond=use_cond)
            acc_list.append(acc)
            loss_test.append(t.item())

    return loss_g_list, loss_d_list, w_distances, loss_test, acc_list
