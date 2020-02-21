from collections import OrderedDict

import torch
import torch.nn as nn
from gym import spaces

from rl.policies.utils import CNN, MLP
from rl.policies.actor_critic import Actor, Critic
from util.gym import observation_size, action_size


class MlpActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy, deterministic=False):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        self._ac_space = ac_space
        self._deterministic = deterministic

        # observation
        input_dim = observation_size(ob_space)

        self.fc = MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size]*2)
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, space in ac_space.spaces.items():
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
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]

        out = self._activation_fn(self.fc(torch.cat(inp, dim=-1)))
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


class MlpCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        input_dim = observation_size(ob_space)
        if ac_space is not None:
            input_dim += action_size(ac_space)

        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2, last_activation=False)

    def forward(self, ob, ac=None):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]

        if ac is not None:
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            inp.extend(ac)

        out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out

