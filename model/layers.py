import os
from collections import namedtuple

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from torch.nn.init import xavier_uniform_

LayerConfig = namedtuple("LayerConfig",
                         ['inDim', 'outDim', 'type', 'activation'])
NetConfig = namedtuple('Netconfig', ['type', 'layers', 'optim', 'loss'])


def build_net(config, output_info_list=None, device='cuda'):
    if config.type == 'MixDataNet':
        return MixDataNet(config=config, output_info_list=output_info_list).to(device)
    elif config.type == 'MLP':
        return MLP(config=config, output_info_list=output_info_list).to(device)
    elif config.type == 'VAE':
        return VAE(config=config).to(device)
    else:
        raise ValueError(f'未知的网络类型：{config.type}')


def build_trainer(config, net):
    if config.optim == 'Adam':
        return torch.optim.Adam(net.parameters())
    else:
        raise ValueError(f'未知的优化器类型{config.optim}')


class BaseModule(nn.Module):
    def __init__(self, param_file_name):
        super().__init__()
        self.param_file_name = param_file_name

    # 读取网络参数
    def load(self, param_path, param_file_name=None):
        if param_file_name is None:
            param_file_name = self.param_file_name
        path = os.path.join(param_path, param_file_name)
        print('loading', path)
        self.load_state_dict(torch.load(path))

    # 保存网络参数
    def save(self, param_path, param_file_name=None):
        if param_file_name is None:
            param_file_name = self.param_file_name
        path = os.path.join(param_path, param_file_name)
        torch.save(self.state_dict(), path)


class MLP(BaseModule):
    def __init__(self, config, output_info_list, param_file_name='default'):
        super().__init__(param_file_name)
        self.config = config
        self.output_info_list = output_info_list
        seq = []
        for l in config.layers:
            seq.append(nn.Linear(l.inDim, l.outDim))
            xavier_uniform_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)
            if l.activation == 'relu':
                seq.append(nn.ReLU())
            elif l.activation == 'sigmoid':
                seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*seq)

    def forward(self, data):
        data = self.seq(data)
        # data = self._activate(data)
        return data

    def _activate(self, data):
        data_out = []
        start = 0
        for column_info in self.output_info_list:
            for info in column_info:
                end = start + info.dim
                if info.activation_fn == 'tanh':
                    # 原CTGAN的为tanh， 但会导致识别器损失消失（值小于0）
                    data_out.append(torch.tanh(data[:, start:end]))
                elif info.activation_fn == 'softmax':
                    data_out.append(gumbel_softmax(data[:, start:end], tau=0.2))
                else:
                    raise ValueError(f'未知的激活函数：{info.activation_fn}')
                start = end
        return torch.cat(data_out, dim=1)


class _Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        xavier_uniform_(self.fc.weight)
        xavier_uniform_(self.fc_mu.weight)
        xavier_uniform_(self.fc_var.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_var.bias)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        z_mu = self.fc_mu(h)
        z_var = self.fc_var(h)
        return z_mu, z_var


class _Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        re_x = torch.sigmoid(self.fc2(h))
        return re_x

class VAE(BaseModule):
    def __init__(self, config, param_file_name='default'):
        super().__init__(param_file_name)
        in_dim = config.layers[0].inDim
        hidden_dim = config.layers[0].outDim
        latent_dim = config.layers[1].inDim
        out_dim = config.layers[1].outDim
        self.encoder = _Encoder(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = _Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):
        mu, var = self.encoder(x)
        z = self._reparameterize(mu, var)
        re_x = self.decoder(z)
        return re_x, mu, var

    def _reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class MixDataNet(BaseModule):
    def __init__(self, config, output_info_list, param_file_name='default'):
        super().__init__(param_file_name)
        self.config = config
        self.output_info_list = output_info_list
        seq = []
        for l in config.layers:
            if l.type == 'Linear':
                seq.append(nn.Linear(l.inDim, l.outDim))
            else:
                raise ValueError(f'未知的层类型：{l.type}')
            xavier_uniform_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)
        self.seq = nn.Sequential(*seq)

    def forward(self, data):
        for f in self.seq:
            data = f(data)
            data = self._activate(data)
        return data

    def _activate(self, data):
        data_out = []
        start = 0
        for column_info in self.output_info_list:
            for info in column_info:
                end = start + info.dim
                if info.activation_fn == 'tanh':
                    # 原CTGAN的为tanh， 但会导致识别器损失消失（值小于0）
                    data_out.append(torch.tanh(data[:, start:end]))
                elif info.activation_fn == 'softmax':
                    data_out.append(gumbel_softmax(data[:, start:end], tau=0.2))
                else:
                    raise ValueError(f'未知的激活函数：{info.activation_fn}')
                start = end
        return torch.cat(data_out, dim=1)
