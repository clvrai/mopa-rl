import os, sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from rl.policies.utils import init

class Decoder(nn.Module):
    def __init__(self, config, input_dim, output_size, hid_dims=[]):
        super().__init__()
        self._config = config
        self.output_size = output_size
        self.activation_fn = nn.ReLU()

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fc[-1].bias.data.fill_(0.1)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, config.rl_hid_size*output_size*output_size))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        self.fc = nn.Sequential(*fc)

        self.deconvs = nn.ModuleList()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                       constant_(x, 0), nn.init.calculate_gain('relu'))

        w = output_size

        deconv_input_dim = config.rl_hid_size
        for k, s, d, p, op in zip(config.decoder_kernel_size,
                           config.decoder_stride, config.decoder_conv_dim, config.decoder_padding, config.decoder_out_padding):
            self.deconvs.append(init_(nn.ConvTranspose2d(deconv_input_dim, d, int(k), int(s), padding=int(p), output_padding=int(op))))
            w = int(int(s) * (w - 1) + int(k) + (int(k) - 1) * (int(d) - 1))
            deconv_input_dim = d


        self.outputs = OrderedDict()


    def forward(self, h):
        config = self._config
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        out = h.view(-1, config.rl_hid_size, self.output_size, self.output_size)

        self.outputs['deconv1'] = out

        for i, deconv in enumerate(self.deconvs[:-1]):
            out = self.activation_fn(deconv(out))
            self.outputs['deconv%s' % (i+1)] = out

        obs = self.deconvs[-1](out)
        self.outputs['obs'] = obs
        return obs

