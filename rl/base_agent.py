from gym import spaces
import numpy as np


class BaseAgent(object):
    def __init__(self, config, ob_space):
        self._config = config

    def normalize(self, ob):
        if self._config.ob_norm:
            return self._ob_norm.normalize(ob)
        return ob

    def act(self, ob, is_train=True, return_stds=False, random_exploration=False):
        if random_exploration:
            ac = self._ac_space.sample()
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Discrete):
                    ac[k] = np.array([ac[k]])
            activation = None
            stds = None
            return ac, activation, stds

        if return_stds:
            ac, activation, stds = self._actor.act(
                ob, is_train=is_train, return_stds=return_stds
            )
            return ac, activation, stds
        else:
            ac, activation = self._actor.act(
                ob, is_train=is_train, return_stds=return_stds
            )
            return ac, activation

    def store_episode(self, rollouts):
        raise NotImplementedError()

    def replay_buffer(self):
        return self._buffer.state_dict()

    def load_replay_buffer(self, state_dict):
        self._buffer.load_state_dict(state_dict)

    def sync_networks(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def _soft_update_target_network(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

    def _copy_target_network(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
