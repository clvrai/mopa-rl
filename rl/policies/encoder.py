import os, sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from rl.policies.utils import init

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class Encoder(nn.Module):
    def __init__(self, config, input_dim, output_dim, hid_dims=[]):
        super().__init__()
        self.activation_fn = nn.ReLU()

        self._config = config

        self.convs = nn.ModuleList()
        w = config.img_width

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), nn.init.calculate_gain('relu'))

        for k, s, d in zip(config.encoder_kernel_size,
                           config.encoder_stride, config.encoder_conv_dim):
            self.convs.append(init_(nn.Conv2d(input_dim, d, int(k), int(s))))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            input_dim = d

        # screen_width == 32 (8,4)-(3,2) -> 3x3
        # screen_width == 64 (8,4)-(3,2)-(3,2) -> 3x3
        # screen_width == 128 (8,4)-(3,2)-(3,2)-(3,2) -> 3x3
        # screen_width == 256 (8,4)-(3,2)-(3,2)-(3,2) -> 7x7

        print('Encoder Output of CNN = %d x %d x %d' % (w, w, d))
        self.w = w
        self.output_size = w * w * d

        fc = []
        prev_dim = self.output_size
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fc[-1].bias.data.fill_(0.1)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        self.fc = nn.Sequential(*fc)
        self.ln = nn.LayerNorm(output_dim)
        self.outputs = OrderedDict()


    def forward(self, ob, detach=False):
        out = ob
        if len(out.shape) == 3:
            out = out.unsqueeze(0)
        for conv in self.convs:
            out = self.activation_fn(conv(out))
        h = out.flatten(start_dim=1)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps*std

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(len(self.convs)):
            tie_weights(src=source.convs[i], trg=self.convs[i])




