from collections import namedtuple

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.functional import gumbel_softmax
from model.base import BaseModule

LayerConfig = namedtuple("LayerConfig",
                         ['inDim', 'outDim', 'activation'])

class MLP(BaseModule):
    def __init__(self, layers, param_file_name='mlp'):
        super().__init__(param_file_name)
        seq = []
        for l in layers:
            seq.append(nn.Linear(l.inDim, l.outDim))
            xavier_uniform_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)
            if l.activation == 'relu':
                seq.append(nn.ReLU())
            elif l.activation == 'sigmoid':
                seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.seq(x)
        return x


class MLP_SA(BaseModule):
    def __init__(self, layers, output_info_list, activate_all, is_gain=True, param_file_name='mlp_sa'):
        super().__init__(param_file_name)
        self.output_info_list = output_info_list
        self.activate_all = activate_all
        self.is_gain = is_gain
        seq = []
        for l in layers:
            seq.append(nn.Linear(l.inDim, l.outDim))
            xavier_uniform_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)
            if activate_all is False:
                if l.activation == 'relu':
                    seq.append(nn.ReLU())
                elif l.activation == 'sigmoid':
                    seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        if self.activate_all:
            for f in self.seq:
                x = f(x)
                x = self._activate(x)
        else:
            x = self.seq(x)
            x = self._activate(x)
        return x

    def _activate(self, x):
        data_out = []
        start = 0
        for column_info in self.output_info_list:
            if len(column_info) == 1:
                n = 1
            else:
                n = 2
            for i in range(n):
                info = column_info[i]
                end = start + info.dim
                if info.activation_fn == 'tanh':
                    # 原CTGAN的为tanh， 但会导致识别器损失消失（值小于0）
                    if self.is_gain:
                        data_out.append(torch.sigmoid(x[:, start:end]))
                    else:
                        data_out.append(torch.tanh(x[:, start:end]))
                elif info.activation_fn == 'softmax':
                    data_out.append(gumbel_softmax(x[:, start:end], tau=0.2))
                else:
                    raise ValueError(f'未知的激活函数：{info.activation_fn}')
                start = end
        return torch.cat(data_out, dim=1)
