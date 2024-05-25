import torch
from torch import nn, autograd


class WDLoss(nn.Module):
    def __init__(self, discriminator, lambda_=10):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def forward(self, real_data, fake_data):
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data)
        grad_penalty = self._calc_gradient_penalty(real_data, fake_data)
        w_distance = torch.mean(d_real) - torch.mean(d_fake)
        return grad_penalty - w_distance, w_distance

    def _calc_gradient_penalty(self, real_data, fake_data, device='cuda', pac=1, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        # alpha为随机生成的插值矩阵，通过下面计算组合real_data和fake_data
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        # nn.Module是callable的，self()是执行forward操作
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

class GAINLoss(nn.Module):
    """
    MSE和交叉熵损失的组合，GAIN generator的损失
    """
    def __init__(self, discriminator, generator, alpha):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.alpha = alpha

    def forward(self, data, mask, hint):
        input_g = torch.cat(dim=1, tensors=[data, mask])
        imputed = self.generator(input_g)
        imputed = imputed * (1-mask) + data * mask
        input_d = torch.cat(dim=1, tensors=[imputed, hint])
        estimate = self.discriminator(input_d)

        loss_train = torch.mean(((1-mask) * data - (1-mask) * imputed) ** 2) / torch.mean(1-mask)
        loss_g = -torch.mean((1 - mask) * torch.log(estimate + 1e-8))
        loss_g += self.alpha * loss_train
        return loss_g


class CTGANLoss(nn.Module):
    def __init__(self, generator, discriminator, output_info_list):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.output_info_list = output_info_list

    def forward(self, data, mask, hint, cond, mask_cond):
        input_g = torch.cat(dim=1, tensors=[data, mask, cond])
        imputed = self.generator(input_g)
        imputed = imputed * (1 - mask) + data * mask
        loss_g = self._cond_loss(imputed, cond, mask_cond)
        input_d = torch.cat(dim=1, tensors=[imputed, hint, cond])
        estimate = self.discriminator(input_d)
        loss_g += -torch.mean(estimate)
        return loss_g

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        for column_info in self.output_info_list:
            for info in column_info:
                if len(column_info) != 1 or info.activation_fn != 'softmax':
                    # 非离散数据不计算交叉熵损失
                    st += info.dim
                else:
                    ed = st + info.dim
                    ed_c = st_c + info.dim
                    tmp = nn.functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

class InformationLoss(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, data, mask, hint, cond):
        input_g = torch.cat(dim=1, tensors=[data, mask, cond])
        imputed = self.generator(input_g)
        imputed = imputed * (1-mask) + data * mask
        input_d_fake = torch.cat(dim=1, tensors=[imputed, hint, cond])
        input_d_real = torch.cat(dim=1, tensors=[data, hint, cond])
        estimate_fake = self.discriminator(input_d_fake)
        estimate_real = self.discriminator(input_d_real)
        l_inform = self._information_loss(estimate_real=estimate_real, estimate_fake=estimate_fake)
        return l_inform

    def _downstream_loss(self):
        return

    def _generator_loss(self):
        return

    def _information_loss(self, estimate_fake, estimate_real):
        # 由于GAIN的识别器本身的输出就是向量，因此直接使用
        # 这部分其实有点像GAIN原版的loss
        m = (torch.mean(estimate_real) - torch.mean(estimate_fake)) ** 2
        s = (torch.std(estimate_real) - torch.mean(estimate_fake)) ** 2
        return m + s

