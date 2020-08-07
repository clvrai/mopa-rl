from collections import defaultdict, OrderedDict
from time import time

import numpy as np
from util.gym import observation_size, action_size

class ReplayBuffer:
    """ Replay Buffer. """

    def __init__(self, buffer_size, sample_func, ob_space, ac_space):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._buffers= defaultdict(list)
        self._obs = {k: np.zeros((buffer_size, observation_size(ob_space[k]))) for k in ob_space.spaces.keys()}
        self._obs_next = {k: np.zeros((buffer_size, observation_size(ob_space[k]))) for k in ob_space.spaces.keys()}
        self._actions = {k: np.zeros((buffer_size, action_size(ac_space[k]))) for k in ac_space.spaces.keys()}
        self._rewards = np.zeros((buffer_size, 1))
        self._terminals = np.zeros((buffer_size, 1))

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    def store_episode(self, rollout):
        """ Stores the episode. """
        idx = self._idx = (self._idx + 1) % self._size
        for k in self._obs.keys():
            for i, data in enumerate(rollout['ob']):
                if self._current_size + i > self._size:
                    self._obs[k][idx] = data[k]
                    self._obs_next[k][idx] = data[k]
                else:
                    self._obs[k][self._current_size+i] = data[k]
                    self._obs_next[k][self._current_size+i] = data[k]
        for k in self._actions.keys():
            for i, data in enumerate(rollout['ac']):
                if self._current_size  + i > self._size:
                    self._actions[k][idx] = data[k]
                else:
                    self._actions[k][self._current_size+i] = data[k]

        for i in range(len(rollout['rew'])):
            if self._current_size > self._size:
                self._rewards[idx] = rollout['rew'][i]
                self._terminals[idx] = rollout['done'][i]
                idx = self._idx = (self._idx + 1) % self._size
            else:
                self._rewards[self._current_size+i] = rollout['rew'][i]
                self._terminals[self._current_size+i] = rollout['done'][i]
                self._current_size += 1

    def sample(self, batch_size):
        """ Samples the data from the replay buffer. """
        # sample transitions
        buffers = {'ob': self._obs,
                   'ob_next': self._obs_next,
                   'done': self._terminals,
                   'rew': self._rewards,
                   'ac': self._actions}
        transitions = self._sample_func(buffers, batch_size, self._current_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac'])


class RandomSampler:
    """ Samples a batch of transitions from replay buffer. """
    def sample_func(self, episode_batch, batch_size_in_transitions, size):
        rollout_batch_size = len(episode_batch['done'])
        batch_size = batch_size_in_transitions

        idxs = np.random.randint(0, size, batch_size)
        transitions = {}
        for k in episode_batch.keys():
            if isinstance(episode_batch[k], dict):
                data = {}
                for sub_key, v in episode_batch[k].items():
                    data[sub_key] = v[idxs]
                transitions[k] = data
            else:
                transitions[k] = episode_batch[k][idxs]
        return transitions
