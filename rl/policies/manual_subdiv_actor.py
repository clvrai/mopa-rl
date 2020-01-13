import numpy as np
import torch
import torch.nn as nn

from rl.policies.utils import CNN, MLP
from rl.policies.actor_critic import Actor


class ManualSubdivActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        #assert config.subdiv is not None, 'subdiv is None for manual subdivision'
        if config.subdiv is None:
            clusters = [(list(ob_shape.keys()), list(ac_space.keys()))]
        else:
            clusters = config.subdiv.split('/')
            clusters = [
                (cluster.split('-')[0].split(','), cluster.split('-')[1].split(',')) for cluster in clusters
            ]

        self._ac_space = ac_space
        self._clusters = clusters

        self.fc = nn.ModuleList()
        self.fc_mean = nn.ModuleList()
        self.fc_log_std = nn.ModuleList()

        for ob_joints, ac_joints in clusters:
            input_dim = sum(ob_space[k] for k in ob_joints)
            output_dim = sum(self._ac_space.shape[k] for k in ac_joints)
            if self._config.diayn:
                input_dim += config.z_dim
            self.fc.append(MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size]))
            self.fc_mean.append(MLP(config, config.rl_hid_size, output_dim))
            self.fc_log_std.append(MLP(config, config.rl_hid_size, output_dim))

    def forward(self, ob):
        means, stds = {}, {}
        for i, (ob_joints, ac_joints) in enumerate(self._clusters):
            ob_keys = ob_joints + (['z'] if self._config.diayn else [])

            inp = torch.cat([ob[k] for k in ob_keys], -1)
            if len(inp.shape) == 1:
                inp = inp.unsqueeze(0)
            out = self._activation_fn(self.fc[i](inp))
            out = torch.reshape(out, (out.shape[0], -1))
            mean = self.fc_mean[i](out)
            log_std = self.fc_log_std[i](out)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)

            prev_ac = 0
            for k in ac_joints:
                next_ac = prev_ac + self._ac_space.shape[k]
                means[k] = mean[:, prev_ac:next_ac]
                stds[k] = std[:, prev_ac:next_ac]
                prev_ac = next_ac

        return means, stds

