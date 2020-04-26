from collections import defaultdict, Counter, OrderedDict
from time import time

import numpy as np


class ReplayBuffer:
    def __init__(self, keys, buffer_size, sample_func):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac'])


class LowLevelReplayBuffer:
    def __init__(self, keys, buffer_size, num_primitives, sample_func):
        self._size = buffer_size
        self._num_primitives = num_primitives

        # memory management
        self._idx = np.zeros(self._num_primitives, dtype=int)
        self._current_size = np.zeros(self._num_primitives)
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = [defaultdict(list) for _ in range(self._num_primitives)]

    def clear(self):
        self._idx = np.zeros(self._num_primitives, dtype=int)
        self._current_size = np.zeros(self._num_primitives)
        self._buffers = [defaultdict(list) for _ in range(self._num_primitives)]

    # store the episode
    def store_episode(self, rollout):
        skill_idx = int(rollout['meta_ac'][0]['default'][0])
        idx = self._idx[skill_idx] = (self._idx[skill_idx] + 1) % self._size
        self._current_size[skill_idx] += 1

        if self._current_size[skill_idx] > self._size:
            for k in self._keys:
                self._buffers[skill_idx][k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[skill_idx][k].append(rollout[k])

    # sample the data from the replay buffer
    def sample(self, batch_size, skill_idx):
        # sample transitions
        buffer = self._buffers[skill_idx]
        transitions = self._sample_func(buffer, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        current_size = []
        for buffer in self._buffers:
            current_size.append(len(buffer['ac']))
        self._current_size = np.array(current_size)

    def create_empty_transition(self):
        transitions = {}
        for key in self._keys:
            transitions[key] = []

        transitions['ob_next'] = []
        return transitions

class LowLevelPPOReplayBuffer:
    def __init__(self, keys, buffer_size, num_primitives, sample_func):
        self._size = buffer_size
        self._num_primitives = num_primitives

        # memory management
        self._idx = np.zeros(self._num_primitives, dtype=int)
        self._current_size = np.zeros(self._num_primitives)
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = [defaultdict(list) for _ in range(self._num_primitives)]

    def clear(self):
        self._idx = np.zeros(self._num_primitives, dtype=int)
        self._current_size = np.zeros(self._num_primitives)
        self._buffers = [defaultdict(list) for _ in range(self._num_primitives)]

    # store the episode
    def store_episode(self, rollout):
        # rollout['ob_next'] = rollout['ob'][1:]
        # meta_ac = np.array([int(d['default']) for d in rollout['meta_ac']])
        sorted_rollout = []
        for i in range(self._num_primitives):
            sorted_rollout.append({})
            for key in rollout.keys():
                sorted_rollout[i][key] = []
            sorted_rollout[i]['ob_next'] = []

        for t in range(len(rollout['ac'])):
            skill_idx = int(rollout['meta_ac'][t]['default'])
            for key in rollout.keys():
                sorted_rollout[skill_idx][key].append(rollout[key][t])
            sorted_rollout[skill_idx]['ob_next'].append(rollout['ob'][t+1])


        if self._current_size[skill_idx] > self._size:
            for skill_idx in range(self._num_primitives):
                if len(sorted_rollout[skill_idx]['ac']) > 0:
                    idx = self._idx[skill_idx] = (self._idx[skill_idx] + 1) % self._size
                    self._current_size[skill_idx] += 1
                for k in self._keys:
                    self._buffers[skill_idx][k][idx] = sorted_rollout[skill_idx][k]
                self._buffers[skill_idx]['ob_next'][idx] = sorted_rollout[skill_idx]['ob_next']
        else:
            for skill_idx in range(self._num_primitives):
                if len(sorted_rollout[skill_idx]['ac']) > 0:
                    idx = self._idx[skill_idx] = (self._idx[skill_idx] + 1) % self._size
                    self._current_size[skill_idx] += 1
                for k in self._keys:
                    self._buffers[skill_idx][k].append(sorted_rollout[skill_idx][k])
                self._buffers[skill_idx]['ob_next'].append(sorted_rollout[skill_idx]['ob_next'])


    # sample the data from the replay buffer
    def sample(self, batch_size, skill_idx):
        # sample transitions
        buffer = self._buffers[skill_idx]
        transitions = self._sample_func(buffer, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        current_size = []
        for buffer in self._buffers:
            current_size.append(len(buffer['ac']))
        self._current_size = np.array(current_size)

    def create_empty_transition(self):
        transitions = {}
        for key in self._keys:
            transitions[key] = []

        transitions['ob_next'] = []
        return transitions

class RandomSampler:
    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch['ac'])
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [np.random.randint(len(episode_batch['ac'][episode_idx])) for episode_idx in episode_idxs]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = \
                [episode_batch[key][episode_idx][t] for episode_idx, t in zip(episode_idxs, t_samples)]

        if 'ob_next' not in episode_batch.keys():
            transitions['ob_next'] = [
                episode_batch['ob'][episode_idx][t + 1] for episode_idx, t in zip(episode_idxs, t_samples)]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions


class HERSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1./1+replay_k)
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch['ac'])
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [np.random.randint(len(episode_batch['ac'][episode_idx])) for episode_idx in episode_idxs]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = \
                [episode_batch[key][episode_idx][t] for episode_idx, t in zip(episode_idxs, t_samples)]

        transitions['ob_next'] = [
            episode_batch['ob'][episode_idx][t + 1] for episode_idx, t in zip(episode_idxs, t_samples)]
        transitions['r'] = np.zeros((batch_size, ))

        # hindsight experience replay
        for i, (episode_idx, t) in enumerate(zip(episode_idxs, t_samples)):
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_t = np.random.randint(t + 1, len(episode_batch['ac'][episode_idx]) + 1)
                future_ag = episode_batch['ag'][episode_idx][future_t]
                if self.reward_func(episode_batch['ag'][episode_idx][t], future_ag, None) < 0:
                    transitions['g'][i] = future_ag

            transitions['r'][i] = self.reward_func(
                episode_batch['ag'][episode_idx][t + 1], transitions['g'][i], None)

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions

