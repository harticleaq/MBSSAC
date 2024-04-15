import numpy as np
import torch
import torch.nn as nn

from torch.distributions import OneHotCategorical


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=2):
        super().__init__()
         # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    

class AttentionEncoder(nn.Module):
    def __init__(self, n_layers, in_dim, hidden, dropout=0.):
        super().__init__()
        self.pos_embed = PositionalEncoding(hidden, 30)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_dim, nhead=8,
                                                                        dim_feedforward=hidden,
                                                                        dropout=dropout), n_layers)

    def forward(self, enc_input, **kwargs):
        enc_input = self.pos_embed(enc_input)
        x = self.encoder(enc_input.permute(1, 0, 2), **kwargs)
        return x.permute(1, 0, 2)