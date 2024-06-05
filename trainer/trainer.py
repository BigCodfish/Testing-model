import torch.optim
from torch import nn
from tqdm import tqdm

from evaluator import Evaluator
from model.loss import WDLoss, CTGANLoss, InformationLoss
from model._vae import VanillaVAE
from utils import utils
from utils.utils import Logger


def _compute_loss_g(generator, discriminator, loss_fn, x, new_x, m, hint, c=None, m_c=None, data_info=None):
    if c is not None:
        input_g = torch.cat(dim=1, tensors=[new_x, m, c])
    else:
        input_g = torch.cat(dim=1, tensors=[new_x, m])
    if loss_fn == 'VAE':
        imputed_x, mu, var = generator(input_g)
    else:
        imputed_x = generator(input_g)
    imputed_x = new_x * m + imputed_x * (1 - m)
    if c is not None:
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint, c])
    else:
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint])
    estimate = discriminator(input_d)

    # 计算损失
    if loss_fn == 'GAIN':
        alpha = 10
        # 尽可能不偏离已知数据
        mse = torch.mean((m * new_x - m * imputed_x) ** 2) / torch.mean(m)
        # 尽可能填补缺失数据
        loss_g = -torch.mean((1 - m) * torch.log(estimate + 1e-8))
        loss_g = loss_g + mse * alpha
    elif loss_fn == 'CTGAN':
        loss_fn = CTGANLoss(discriminator=discriminator, generator=generator, output_info_list=data_info)
        loss_g = loss_fn(new_x, m, hint, c, m_c)
    elif loss_fn == 'Information':
        loss_fn = InformationLoss(generator=generator, discriminator=discriminator)
        loss_g = loss_fn(data=new_x, mask=m, hint=hint, cond=c)
    elif loss_fn == 'VAE':
        ce = nn.functional.cross_entropy(imputed_x, x, reduction='mean')
        kld = torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))
        loss_g = ce + kld
        loss_g += torch.mean((1 - m) * torch.log(estimate + 1e-8))
    else:
        raise ValueError(f'Unknown generator loss type: {loss_fn}')
    return loss_g, imputed_x


def _compute_loss_d(generator, discriminator, loss_fn, x, new_x, m, hint, c=None):
    if c is not None:
        input_g = torch.cat(dim=1, tensors=[new_x, m, c])
    else:
        input_g = torch.cat(dim=1, tensors=[new_x, m])
    if isinstance(generator, VanillaVAE):
        imputed_x,_,_ = generator(input_g)
    else:
        imputed_x = generator(input_g)
    imputed_x = new_x * m + imputed_x * (1 - m)
    if c is not None:
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint, c])
        input_d_real = torch.cat(dim=1, tensors=[x, m, c])
    else:
        input_d = torch.cat(dim=1, tensors=[imputed_x, hint])
        input_d_real = torch.cat(dim=1, tensors=[x, m])
    compute_wd = WDLoss(discriminator)
    estimate = discriminator(input_d)
    wd_penalty, wd = compute_wd(input_d_real, input_d)
    ce = -torch.mean(m * torch.log(estimate + 1e-8) + (1 - m) * torch.log(1. - estimate + 1e-8))
    if loss_fn == 'GAIN':
        loss_d = ce
    elif loss_fn == 'WD':
        loss_d = wd_penalty
    else:
        raise ValueError(f'Unknown discriminator loss type: {loss_fn}')
    return loss_d, wd, ce


def train(train_x, test_x, generator, discriminator, loss_fn, data_info=None, data_sampler=None, d_token=-1, decoder=None,
                rate_0=0.2, hint_rate=0.8,
                batch_size=128, num_epochs=1000, lr=1e-3, device='cuda'):
    train_m, test_m = utils.generate_mask(train_x, test_x, rate_0, data_info=data_info, d_token=d_token)
    dim = train_x.shape[-1]
    print(f'samples count: {len(train_x)}, sample dim: {dim}, loss_g: {loss_fn[0]}, loss_d: {loss_fn[1]}')
    evaluator = Evaluator(test_x, test_m, rate_0, d_token, decoder, data_info, device)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    trainer_g = torch.optim.Adam(generator.parameters())
    trainer_d = torch.optim.Adam(discriminator.parameters())

    logger = Logger(['loss_g', 'loss_d', 'train_wd', 'train_ce_d', 'train_mse', 'train_acc', 'test_mse', 'test_acc'])
    iterator = tqdm(range(num_epochs))
    for epoch in iterator:
        idx = utils.sample_idx(len(train_x), batch_size)
        raw_x = train_x[idx, :]
        m = train_m[idx, :]
        # 是否使用原始数据参与训练
        x = m * raw_x
        z = utils.generate_noise(batch_size, dim)
        new_x = x * m + z * (1 - m)
        hint = utils.generate_mask(m, None, 1 - hint_rate, data_info=data_info, d_token=d_token)
        hint = hint * m + 0.5 * (1-hint) * m

        if data_sampler is not None:
            c, m_c, i, j = data_sampler.sample_cond(batch_size)
            c = torch.tensor(c, dtype=torch.float32, device='cuda')
            m_c = torch.tensor(m_c, dtype=torch.float32, device='cuda')
        else:
            c, m_c, i, j = None, None, None, None
        x = torch.tensor(x, dtype=torch.float32, device=device)
        m = torch.tensor(m, dtype=torch.float32, device=device)
        hint = torch.tensor(hint, dtype=torch.float32, device=device)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=device)
        raw_x = torch.tensor(raw_x, dtype=torch.float32, device=device)

        trainer_d.zero_grad()
        loss_d, wd, ce = _compute_loss_d(generator, discriminator, loss_fn[1], x, new_x, m, hint, c)
        loss_d.backward()
        trainer_d.step()

        trainer_g.zero_grad()
        loss_g, imputed_x = _compute_loss_g(generator, discriminator, loss_fn[0], x, new_x, m, hint, c, m_c, data_info)
        loss_g.backward()
        trainer_g.step()

        '-- Evaluation --'
        if epoch % 10 == 0:
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                if data_info is not None:
                    train_mse = evaluator.compute_mse_msn(imputed_x, raw_x)
                    train_acc = evaluator.compute_acc_msn(imputed_x, raw_x)
                    test_mse, test_acc = evaluator.eval_msn(generator, data_sampler)
                else:
                    imputed_x = imputed_x.view(imputed_x.shape[0], -1, d_token)
                    raw_x = raw_x.view(imputed_x.shape[0], -1, d_token)
                    i_x_num, i_x_cat = decoder(imputed_x[:, 1:])
                    x_num, x_cat = decoder(raw_x[:, 1:])
                    total = (x_num.shape[0] * x_num.shape[1]) * rate_0
                    train_mse = nn.functional.mse_loss(x_num, i_x_num, reduction='sum') / total
                    train_acc = evaluator.compute_acc_vae(i_x_cat, x_cat)
                    test_mse, test_acc = evaluator.eval_vae(generator)
                logger.append([loss_g.item(), loss_d.item(),
                               wd.item(), ce.item(), train_mse.item(), train_acc, test_mse.item(), test_acc])
            if epoch % 50 == 0:
                print("epoch: {}, Train acc = {:.3f}, Train mse = {:.3f}, Test acc = {:.3f}, Test mse = {:.3f}".format(
                      epoch, train_acc, train_mse.item(), test_acc, test_mse.item()))
            generator.train()
            discriminator.train()

    return logger

