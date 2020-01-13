from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.policies.distributions import FixedCategorical, FixedNormal, \
    MixedDistribution
from rl.policies.utils import MLP
from util.pytorch import to_tensor


class Actor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__()
        self._config = config
        self._activation_fn = getattr(F, config.rl_activation)
        self._tanh = tanh_policy

        # modules for DIAYN
        diayn = [k for k in ob_space.keys() if '_diayn' in k]
        if diayn:
            self.z_dim = config.z_dim
            self.z_dist = config.z_dist
            self._sampled_z = None
            assert len(diayn) == 1
            self.z_name = diayn[0]

            # discriminator q(z|s)
            if self.z_dist == 'normal':
                output_dim = self.z_dim * 2
            elif self.z_dist == 'categorical':
                output_dim = self.z_dim
            else:
                raise ValueError('z distribution {} is undefined.'.format(self.z_dist))
            input_dim = sum(ob_space.values()) - config.z_dim
            self.discriminator = MLP(config, input_dim,
                                     output_dim,
                                     [config.rl_hid_size] * 2)
        self.diayn = diayn

    @property
    def info(self):
        return {}

    def act(self, ob, is_train=True, return_log_prob=False):
        ob = to_tensor(ob, self._config.device)
        self._ob = ob
        means, stds = self.forward(ob)

        dists = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        actions = OrderedDict()
        mixed_dist = MixedDistribution(dists)
        if not is_train:
            activations = mixed_dist.mode()
        else:
            activations = mixed_dist.sample()

        if return_log_prob:
            log_probs = mixed_dist.log_probs(activations)

        for k in self._ac_space.keys():
            z = activations[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                if return_log_prob:
                    # follow the Appendix C. Enforcing Action Bounds
                    log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                    log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action.detach().cpu().numpy().squeeze(0)
            activations[k] = z.detach().cpu().numpy().squeeze(0)

        if return_log_prob:
            log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
            if log_probs_.min() < -100:
                print('sampling an action with a probability of 1e-100')
                import ipdb; ipdb.set_trace()

            log_probs_ = log_probs_.detach().cpu().numpy().squeeze(0)
            return actions, activations, log_probs_
        else:
            return actions, activations

    def act_log(self, ob, activations=None):
        self._ob = ob.copy()
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample() if activations is None else activations
        for k in activations_.keys():
            if len(activations_[k].shape) == 1:
                activations_[k] = activations_[k].unsqueeze(0)
        log_probs = mixed_dist.log_probs(activations_)

        for k in self._ac_space.keys():
            z = activations_[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        ents = mixed_dist.entropy()
        log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
        if activations is None:
            return actions, log_probs_
        else:
            return log_probs_, ents

    def act_log_debug(self, ob, activations=None):
        means, stds = self.forward(ob)

        dists = OrderedDict()
        actions = OrderedDict()
        for k in self._ac_space.keys():
            if self._ac_space.is_continuous(k):
                dists[k] = FixedNormal(means[k], stds[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        mixed_dist = MixedDistribution(dists)

        activations_ = mixed_dist.rsample() if activations is None else activations
        log_probs = mixed_dist.log_probs(activations_)

        for k in self._ac_space.keys():
            z = activations_[k]
            if self._tanh and self._ac_space.is_continuous(k):
                action = torch.tanh(z)
                # follow the Appendix C. Enforcing Action Bounds
                log_det_jacobian = 2 * (np.log(2.) - z - F.softplus(-2. * z)).sum(dim=-1, keepdim=True)
                log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        ents = mixed_dist.entropy()
        #print(torch.cat(list(log_probs.values()), -1))
        log_probs_ = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
        if log_probs_.min() < -100:
            print(ob)
            print(log_probs_.min())
            import ipdb; ipdb.set_trace()
        if activations is None:
            return actions, log_probs_
        else:
            return log_probs_, ents, log_probs, means, stds

    def custom_loss(self):
        if self.diayn:
            # get discriminator output
            inp = [self._ob[k] for k in self._ob if '_diayn' not in k] # list(self._ob.values())
            if len(inp[0].shape) == 1:
                inp = [x.unsqueeze(0) for x in inp]
            q_output = self.discriminator(torch.cat(inp, dim=-1))

            sampled_z = self._ob[self.z_name]
            if self.z_dist == 'normal':
                mean, log_std = torch.chunk(q_output, 2, dim=-1)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
                output_dist = FixedNormal(mean, std)
                normal_dist = FixedNormal(torch.zeros_like(mean), torch.ones_like(std))
                discriminator_log_probs = output_dist.log_probs(sampled_z)
                discriminator_log_probs = torch.clamp(discriminator_log_probs, -20, 20)
                normal_log_probs = normal_dist.log_probs(sampled_z)
                normal_log_probs = torch.clamp(normal_log_probs, -20, 20)
                discriminator_loss = -discriminator_log_probs.mean() + \
                    normal_log_probs.mean()

            else:
                softmax_ce_w_logits = lambda labels, logits: torch.sum(-labels * F.log_softmax(logits, -1), -1).mean()
                discriminator_loss = softmax_ce_w_logits(sampled_z, q_output)
            return discriminator_loss
        else:
            return None

    '''
    Methods for DIAYN
    '''

    def _sample_z(self):
        if self.z_dist == 'normal':
            self._sampled_z = np.random.normal(0, 1, (self.z_dim,))
        else:
            z_value = np.random.choice(self.z_dim)
            self._sampled_z = np.zeros((self.z_dim,), dtype=np.float32)
            self._sampled_z[z_value] = 1
        return {self.z_name: self._sampled_z}

    '''
    def _append_z(self, ob, z=None):
        ob_shape_sample = list(self._ob.values())[0].shape
        ob['z'] = self._sampled_z if z is None else z
        ob_size = len(ob_shape_sample)
        z_size = len(z.shape)
        if len(ob_shape_sample) == 1:
            batch_size = 1
        else:
            batch_size = ob_shape_sample[0]
        if ob_size > z_size:
            ob['z'] = np.array([ob['z']] * batch_size)
        return ob
    '''


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

