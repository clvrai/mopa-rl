import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.activation_fn = nn.ReLU()

        self.convs = nn.ModuleList()
        w = config.img_width
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        for k, s, d in zip(config.kernel_size, config.stride, config.conv_dim):
            self.convs.append(init_(nn.Conv2d(input_dim, d, int(k), int(s))))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            input_dim = d

        # screen_width == 32 (8,4)-(3,2) -> 3x3
        # screen_width == 64 (8,4)-(3,2)-(3,2) -> 3x3
        # screen_width == 128 (8,4)-(3,2)-(3,2)-(3,2) -> 3x3
        # screen_width == 256 (8,4)-(3,2)-(3,2)-(3,2) -> 7x7

        print("Output of CNN = %d x %d x %d" % (w, w, d))
        self.w = w
        self.output_size = w * w * d

    def forward(self, ob):
        out = ob
        for conv in self.convs:
            out = self.activation_fn(conv(out))
        out = out.flatten(start_dim=1)
        return out


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MLP(nn.Module):
    def __init__(
        self,
        config,
        input_dim,
        output_dim,
        hid_dims=[],
        last_activation=False,
        activation="relu",
    ):
        super().__init__()
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        elif acitvation == "elu":
            activation_fn = nn.Elu()
        else:
            return NotImplementedError

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fanin_init(fc[-1].weight)
            fc[-1].bias.data.fill_(0.1)
            fc.append(activation_fn)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        if last_activation:
            fc.append(activation_fn)
        self.fc = nn.Sequential(*fc)

    def forward(self, ob):
        return self.fc(ob)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
