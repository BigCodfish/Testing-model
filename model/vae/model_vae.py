import math
import os
import sys

import torch
from torch import nn, Tensor

from model.layers import BaseModule

class MLP_VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class VAE(BaseModule):
    def __init__(self, dim_num, categories, dim_token, in_dim, hidden_dim, latent_dim, model_file_name='vae'):
        super().__init__(model_file_name)
        self.encoder = Encoder(dim_num, categories, dim_token, in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(dim_num, categories, dim_token, in_dim, hidden_dim, latent_dim)

    def forward(self, x_num, x_cat):
        mu, var = self.encoder(x_num, x_cat)
        z = self._reparameterize(mu, var)
        re_x_num, re_x_cat = self.decoder(z)
        return re_x_num, re_x_cat, mu, var

    def _reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class Encoder(BaseModule):
    def __init__(self, dim_num, categories, dim_token, in_dim, hidden_dim, latent_dim, model_file_name='encoder'):
        super().__init__(model_file_name)
        self.tokenizer = Tokenizer(dim_num=dim_num, categories=categories, dim_token=dim_token, bias=True)
        self.encode_mu = MLP_VAE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=latent_dim)
        self.encode_var = MLP_VAE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=latent_dim)

    def forward(self, x_num, x_cat):
        h = self.tokenizer(x_num, x_cat)
        h = h.view(h.shape[0], -1)
        mu = self.encode_mu(h)
        var = self.encode_var(h)
        return mu, var


class Decoder(BaseModule):
    def __init__(self, dim_num, categories, dim_token, in_dim, hidden_dim, latent_dim, model_file_name='encoder'):
        super().__init__(model_file_name)
        self.decode = MLP_VAE(in_dim=latent_dim, hidden_dim=hidden_dim, out_dim=in_dim)
        self.reconstructor = Reconstructor(dim_num=dim_num, categories=categories, dim_token=dim_token)

    def forward(self, z):
        re_x = torch.sigmoid(self.decode(z))
        re_x_num, re_x_cat = self.reconstructor(re_x)
        return re_x_num, re_x_cat


class Tokenizer(nn.Module):
    def __init__(self, dim_num, categories, dim_token, bias):
        super().__init__()
        if categories is None:
            d_bias = dim_num
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = dim_num + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), dim_token)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(dim_num + 1, dim_token))
        self.bias = nn.Parameter(Tensor(d_bias, dim_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device='cuda')]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x


class Reconstructor(nn.Module):
    def __init__(self, dim_num, categories, dim_token):
        super(Reconstructor, self).__init__()

        self.dim_num = dim_num
        self.categories = categories
        self.dim_token = dim_token

        self.weight = nn.Parameter(Tensor(dim_num, dim_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(dim_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h = h.view(h.shape[0], -1, self.dim_token)
        h_num = h[:, :self.dim_num]
        h_cat = h[:, self.dim_num:]

        re_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        re_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            re_x_cat.append(recon(h_cat[:, i]))

        return re_x_num, re_x_cat
