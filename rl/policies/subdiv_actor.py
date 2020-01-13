import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.utils import CNN, MLP
from rl.policies.actor_critic import Actor
from rl.policies.distributions import FixedCategorical


class SubdivActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        input_dim = sum(ob_space.values())
        self._num_subpolicy = len(ac_space.keys())
        self.hard_ob_mask = config.hard_ob_mask
        self._ac_space = ac_space

        if self.hard_ob_mask:
            self.ob_masks = nn.Parameter(
                torch.randn(self._num_subpolicy, input_dim, device=self._config.device, requires_grad=True)
            )
        else:
            self.ob_masks = nn.ModuleList()
            for ac_key in ac_space.shape.keys():
                self.ob_masks.append(MLP(config, input_dim, config.ob_mask_size, [config.rl_hid_size]))

        self.fc = nn.ModuleList()
        self.fc_mean = nn.ModuleList()
        self.fc_log_std = nn.ModuleList()

        for ac_key in ac_space.shape.keys():
            ac_size = ac_space.shape[ac_key]
            policy_input_dim = input_dim if self.hard_ob_mask else config.ob_mask_size
            self.fc.append(MLP(config, policy_input_dim, config.rl_hid_size, [config.rl_hid_size]))
            self.fc_mean.append(MLP(config, config.rl_hid_size, ac_size))
            self.fc_log_std.append(MLP(config, config.rl_hid_size, ac_size))

    def forward(self, ob):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]
        inp = torch.cat(inp, dim=-1)

        means, stds = {}, {}
        for i, ac_joint in enumerate(self._ac_space.keys()):
            # mask observation by observation mask
            if self.hard_ob_mask:
                mask = torch.clamp(torch.sigmoid(self.ob_masks[i]), min=0)
                self.masked_ob = inp * mask.unsqueeze(0)
            else:
                self.masked_ob = self.ob_masks[i](inp)

            # compute mean and std
            out = self._activation_fn(self.fc[i](self.masked_ob))
            out = torch.reshape(out, (out.shape[0], -1))
            mean = self.fc_mean[i](out)
            log_std = self.fc_log_std[i](out)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)

            means[k] = mean
            stds[k] = std

        return means, stds

    def custom_loss(self):
        if self.hard_ob_mask:
            if self._config.custom_loss_type == 'ent':
                dist = FixedCategorical(F.softmax(torch.sigmoid(self.ob_masks), dim=-1))
                ent = dist.entropy()
                return ent
            elif self._config.custom_loss_type == 'mean':
                return torch.mean(torch.sigmoid(self.ob_masks))
            else:
                raise NotImplementedError()
        else:
            return torch.sum(torch.mul(self.masked_ob, self.masked_ob))

    @property
    def info(self):
        if self.hard_ob_mask:
            ob_mask_raw = self.ob_masks.detach().cpu().numpy()
            ob_mask_img = np.expand_dims(ob_mask_raw, axis=-1)
            ob_mask_img = np.repeat(ob_mask_img, 20, axis=0)
            ob_mask_img = np.repeat(ob_mask_img, 20, axis=1)
            return { 'ob_masks': ob_mask_img }
        else:
            return {}

