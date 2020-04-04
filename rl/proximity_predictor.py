import os, sys
import torch
import torch.nn as nn

from util.logger import logger


class Replay(object):
    def __init__(self, max_size=1e6, name='buffer'):
        self.max_size = max_size
        self.obs = deque(maxlen=max_size)
        self.collected_obs = []
        self.name = name

    def size(self):
        return len(self.obs) + len(self.collected_obs)

    def list(self):
        return list(self.obs) + self.collected_obs

    def add_collected_obs(self, ob):
        if isinstance(ob, list):
            self.collected_obs.extend(ob)
        elif isinstance(ob, np.ndarray):
            if len(ob.shape) == 1:
                self.collected_obs.append(ob)
            else:
                for i in range(ob.shape[0]):
                    self.collected_obs.append(ob[i, :])
        else:
            self.collected_obs.append(np.ones((1,)) * ob)

    def add(self, ob):
        if isinstance(ob, list):
            self.obs.extend(ob)
        elif isinstance(ob, np.ndarray):
            if len(ob.shape) == 1:
                self.obs.append(ob)
            else:
                for i in range(ob.shape[0]):
                    self.obs.append(ob[i, :])
        else:
            self.obs.append(np.ones((1,)) * ob)

    def sample(self, batchsize):
        idx = []
        idx_collected = []
        if len(self.collected_obs) and len(self.obs):
            idx = np.random.randint(len(self.obs), size=batchsize//2)
            idx_collected = np.random.randint(len(self.collected_obs), size=batchsize//2)
            return np.concatenate([self.get(idx), self.get_collected_obs(idx_collected)], axis=0)
        elif len(self.collected_obs):
            idx_collected = np.random.randint(len(self.collected_obs), size=batchsize)
            return self.get_collected_obs(idx_collected)
        else:
            idx = np.random.randint(len(self.obs), size=batchsize)
            return self.get(idx)

    def get(self, idx):
        return np.stack([self.obs[i] for i in idx])

    def get_collected_obs(self, idx):
        return np.stack([self.collected_obs[i] for i in idx])

    def iterate_times(self, batchsize, times):
        for x in range(times):
            yield self.sample(batchsize)

class ProximityPredictor(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()

        self._config = config
        self.fail_buffer = Replay(
            max_size=config.proximity_replay_size, name='fail_buffer'
        )
        self.success_buffer = Replay(
            max_size=config.proximity_replay_size, name='success_buffer'
        )

        self._num_hidden_layer = config.proximity_num_hidden_layers
        self._hidden_size = config.proximity_hidden_size
        self.activation_fn = nn.ReLU()

        if is_train or config.evaluate_proximity_predictor:
            state_file_path = osp.join(config.primitive_dir, path.split('/')[0], 'state')
            logger.info('Search state files from: {}'.format(config.primitive_dir))
            state_file_list = glob.glob(osp.join(state_file_path, '*.hdf5'))
            logger.info('Candidate state files: {}'.format(
                ' '.join([f.split('/')[-1] for f in state_file_list])))
            state_file = {}
            try:
                logger.info('Use state files: {}'.format(state_file_list[0].split('/')[-1]))
                state_file = h5py.File(state_file_list[0], 'r')
            except:
                logger.warn("No collected state hdf5 file is located at {}".format(
                    state_file_path))
            logger.info('Use traj portion: {} to {}'.format(
                use_traj_portion_start, use_traj_portion_end))

            if self._config.proximity_keep_collected_obs:
                add_obs = self.success_buffer.add_collected_obs
            else:
                add_obs = self.success_buffer.add

            for k in list(state_file.keys()):
                traj_state = state_file[k]['obs'].value
                start_idx = int(traj_state.shape[0]*use_traj_portion_start)
                end_idx = int(traj_state.shape[0]*use_traj_portion_end)
                try:
                    if state_file[k]['success'].value == 1:
                        traj_state = traj_state[start_idx:end_idx]
                    else:
                        continue
                except:
                    traj_state = traj_state[start_idx:end_idx]
                for t in range(traj_state.shape[0]):
                    ob = traj_state[t][:self.observation_shape]
                    # [ob, label]
                    add_obs(np.concatenate((ob, [1.0]), axis=0))

            # shape [num_state, dim_state]
            logger.info('Size of collected state: {}'.format(self.success_buffer.size()))
            logger.info('Average of collected state: {}'.format(np.mean(self.success_buffer.list(), axis=0)))

        fail_fc = []
        success_fc = []
        prev_dim = ob_size
        for _ in range(self._num_hidden_layer):
            fail_fc.append(nn.Linear(prev_dim, self._hidden_size))
            success_fc.append(nn.Linear(prev_dim, self._hidden_size))
            fail_fc.append(self.activation_fn)
            success_fc.append(self.activation_fn)
            prev_dim = self._hidden_size

        fail_fc.append(nn.Linear(prev_dim, 1))
        success_fc.append(nn.Linear(prev_dim, 1))
        self.fail_fc = nn.Sequential(*fc)
        self.success_fc = nn.Sequential(*fc)

    def forward(self, fail_obs, success_obs):
        # fix here
        fail_logits = self.fail_fc(fail_obs)
        success_logits = self.success_fc(success_obs)
        return fail_logits, success_logits

    def _network_cuda(self, device):
        self.fail_fc.to(device)
        self.success_fc.to(device)

    def sync_network(self):
        sync_network(self.fail_fc)
        sync_network(self.success_fc)

    def train(self):
        for i in range(self._config.proximity_num_batches):
            success_transitions = self.success_transitions
        raise NotImplementedError

    def proximity(self, obs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if obs.shape[1] == self.observation_shape:
            obs = np.concatenate((obs, np.zeros(shape=[obs.shape[0], 1])), axis=1)
        success_logits = self.success_fc(obs)
        return torch.clamp(success_logits, 0, 1)

    def _update_network(self, fail_obs, success_obs):
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        fail_obs = _to_tensor(fail_obs)
        success_obs = _to_tensor(success_obs)


        raise NotImplementedError
