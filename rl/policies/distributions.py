from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions


# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)
).unsqueeze(-1)

categorical_entropy = FixedCategorical.entropy
FixedCategorical.entropy = lambda self: categorical_entropy(self) * 10.0  # scaling

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

normal_init = FixedNormal.__init__
FixedNormal.__init__ = lambda self, mean, std: normal_init(
    self, mean.double(), std.double()
)

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = (
    lambda self, actions: log_prob_normal(self, actions.double())
    .sum(-1, keepdim=True)
    .float()
)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1).float()

FixedNormal.mode = lambda self: self.mean.float()

normal_sample = FixedNormal.sample
FixedNormal.sample = lambda self: normal_sample(self).float()

normal_rsample = FixedNormal.rsample
FixedNormal.rsample = lambda self: normal_rsample(self).float()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logstd = AddBias(torch.zeros(config.action_size))
        self.config = config

    def forward(self, x):
        zeros = torch.zeros(x.size()).to(self.config.device)
        logstd = self.logstd(zeros)
        return FixedNormal(x, logstd.exp())


class MixedDistribution(nn.Module):
    def __init__(self, distributions):
        super().__init__()
        assert isinstance(distributions, OrderedDict)
        self.distributions = distributions

    def mode(self):
        return OrderedDict([(k, dist.mode()) for k, dist in self.distributions.items()])

    def sample(self):
        return OrderedDict(
            [(k, dist.sample()) for k, dist in self.distributions.items()]
        )

    def rsample(self):
        return OrderedDict(
            [(k, dist.rsample()) for k, dist in self.distributions.items()]
        )

    def log_probs(self, x):
        assert isinstance(x, dict)
        return OrderedDict(
            [(k, dist.log_probs(x[k])) for k, dist in self.distributions.items()]
        )

    def entropy(self):
        return sum([dist.entropy() for dist in self.distributions.values()])


class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    """
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    """

    def sample(self, sample_shape=torch.Size()):
        """Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical"""
        u = torch.empty(
            self.logits.size(), device=self.logits.device, dtype=self.logits.dtype
        ).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        """
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        """
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        """value is one-hot or relaxed"""
        # if self.logits.shape[-1] == 1:
        #     value = torch.zeros_like(value)
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return -torch.sum(-value * F.log_softmax(self.logits, -1), -1)  # scaling

    def entropy(self):
        return self.base_dist._categorical.entropy()


FixedGumbelSoftmax = GumbelSoftmax
old_sample_gumbel = FixedGumbelSoftmax.sample
FixedGumbelSoftmax.sample = lambda self: old_sample_gumbel(self).unsqueeze(-1)
log_prob_gumbel = FixedGumbelSoftmax.log_prob
FixedGumbelSoftmax.log_probs = lambda self, actions: log_prob_gumbel(
    self, actions.squeeze(-1)
).unsqueeze(-1)
gumbel_entropy = FixedGumbelSoftmax.entropy
FixedGumbelSoftmax.entropy = lambda self: gumbel_entropy(self)
FixedGumbelSoftmax.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
gumbel_rsample = FixedGumbelSoftmax.rsample
FixedGumbelSoftmax.rsample = lambda self: gumbel_rsample(self).float()
