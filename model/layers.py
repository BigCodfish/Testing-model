import os
from collections import namedtuple

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from torch.nn.init import xavier_uniform_

LayerConfig = namedtuple("LayerConfig",
                         ['inDim', 'outDim', 'type', 'activation'])
NetConfig = namedtuple('Netconfig', ['type', 'layers', 'optim', 'loss'])


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
            if l.activation == 'relu':
                seq.append(nn.ReLU())
            elif l.activation == 'sigmoid':
                seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*seq)

    def forward(self, data):
        data = self.seq(data)
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
