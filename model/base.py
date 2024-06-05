import os

import torch
from torch import nn


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