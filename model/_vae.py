import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from model.base import BaseModule


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

class VanillaVAE(BaseModule):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim, param_file_name='default'):
        super().__init__(param_file_name)
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
