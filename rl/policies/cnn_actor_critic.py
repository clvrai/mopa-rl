from collections import OrderedDict

import torch
import torch.nn as nn
from gym import spaces

from rl.policies.utils import CNN, MLP
from rl.policies.actor_critic import Actor, Critic
from util.gym import observation_size, action_size

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class CNNActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy, deterministic=False):
        super().__init__(config, ob_space, ac_space, tanh_policy, deterministic)

        self._ac_space = ac_space
        self._ob_space = ob_space
        self._deterministic = deterministic

        # observation
        # Change this later
        input_shape = ob_space['default'].shape
        input_dim = input_shape[0]

        self.base = CNN(config, input_dim)

        self.aux_fc = nn.ModuleDict()
        out_size = self.base.output_size

        # For basiaclly subgoal
        self._aux_keys = []
        for k, space in self._ob_space.spaces.items():
            if len(space.shape) == 1:
                self.aux_fc.update({k: MLP(config, observation_size(space), int(config.rl_hid_size/4))})
                out_size += config.rl_hid_size/4
                self._aux_keys.append(k)

        self.fc = MLP(config, int(out_size), config.rl_hid_size, [config.rl_hid_size], last_activation=True)
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Box):
                self.fc_means.update({k: MLP(config, config.rl_hid_size, action_size(space))})
                if not self._deterministic:
                    self.fc_log_stds.update({k: MLP(config, config.rl_hid_size, action_size(space))})
            elif isinstance(space, spaces.Discrete):
                self.fc_means.update({k: MLP(config, config.rl_hid_size, space.n)})
            else:
                self.fc_means.update({k: MLP(config, config.rl_hid_size, space)})

    def forward(self, ob, deterministic=False):
        inp = list(ob.values())
        x = ob['default']

        # img process
        if len(x.shape) ==3:
            x = x.unsqueeze(0)

        out = self.base(x)

        # auxiliary feature processing
        aux_feat = []
        for k in self._aux_keys:
            if len(ob[k].shape) == 1:
                ob[k] = ob[k].unsqueeze(0)
            aux_out = self._activation_fn(self.aux_fc[k](ob[k]))
            aux_feat.append(aux_out)
        if len(aux_feat) > 0:
            aux_feat = torch.cat(aux_feat, dim=-1)
            out = torch.cat([out, aux_feat], dim=1)

        out = self._activation_fn(self.fc(out))

        out = torch.reshape(out, (out.shape[0], -1))

        means, stds = OrderedDict(), OrderedDict()

        for k, space in self._ac_space.spaces.items():
            mean = self.fc_means[k](out)
            if isinstance(space, spaces.Box) and not self._deterministic:
                log_std = self.fc_log_stds[k](out)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None

            means[k] = mean
            stds[k] = std
        return means, stds


class CNNCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._activation_fn = nn.ReLU()


        input_shape = ob_space['default'].shape
        input_dim = input_shape[0]

        self.base = CNN(config, input_dim)

        self.aux_fc = nn.ModuleDict()
        out_size = self.base.output_size

        if ac_space is not None:
            out_size += action_size(ac_space)
        # For basiaclly subgoal
        self._aux_keys = []
        for k, space in self._ob_space.spaces.items():
            if len(space.shape) == 1:
                self.aux_fc.update({k: MLP(config, observation_size(space), config.rl_hid_size, [config.rl_hid_size])})
                out_size += config.rl_hid_size
                self._aux_keys.append(k)

        self.fc = MLP(config, out_size, 1, [config.rl_hid_size]*2)


    def forward(self, ob, ac=None):
        inp = list(ob.values())
        x = inp[0]
        # img process
        if len(x.shape) ==3:
            x = x.unsqueeze(0)

        out = self.base(x)

        if ac is not None:
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            out = torch.cat([out, ac[0]], dim=1)


        aux_feat = []
        for k in self._aux_keys:
            if len(ob[k].shape) == 1:
                ob[k] = ob[k].unsqueeze(0)
            aux_out = self._activation_fn(self.aux_fc[k](ob[k]))
            aux_feat.append(aux_out)

        if len(aux_feat) > 0:
            aux_feat = torch.cat(aux_feat, dim=-1)
            out = torch.cat([out, aux_feat], dim=1)

        out = self.fc(out)
        out = torch.reshape(out, (out.shape[0], 1))

        return out

