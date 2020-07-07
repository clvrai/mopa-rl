from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dataset import ReplayBuffer, RandomSampler
from rl.base_agent import BaseAgent
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor
from util.gym import action_size, observation_size
from gym import spaces

class TD3Agent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic, non_limited_idx=None, ref_joint_pos_indexes=None, joint_space=None, is_jnt_limited=None, jnt_indices=None):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._log_alpha = [torch.zeros(1, requires_grad=True, device=config.device)]
        self._alpha_optim = [optim.Adam([self._log_alpha[0]], lr=config.lr_actor)]
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        if joint_space is not None:
            self._jnt_minimum = joint_space['default'].low
            self._jnt_maximum = joint_space['default'].high

        self._build_actor(actor)
        self._critic1 = critic(config, ob_space, ac_space)
        self._critic2 = critic(config, ob_space, ac_space)

        self._critic1_target = critic(config, ob_space, ac_space)
        self._critic2_target = critic(config, ob_space, ac_space)
        self._build_target_actor(actor)
        self._network_cuda(config.device)

        self._actor_optims = [optim.Adam(_actor.parameters(), lr=config.lr_actor) for _actor in self._actors]
        self._critic1_optim = optim.Adam(self._critic1.parameters(), lr=config.lr_critic)
        self._critic2_optim = optim.Adam(self._critic2.parameters(), lr=config.lr_critic)

        sampler = RandomSampler()
        buffer_keys = ['ob', 'ac', 'meta_ac', 'done', 'rew']
        if config.planner_integration:
            buffer_keys.append("intra_steps")
        self._buffer = ReplayBuffer(buffer_keys,
                                    config.buffer_size,
                                    sampler.sample_func)

        self._ounoise = OUNoise(action_size(ac_space))

        self._log_creation()
        self._planner = None
        self._is_planner_initialized = False
        if config.planner_integration:
            self._planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.planner_type,
                                         passive_joint_idx=config.passive_joint_idx, ignored_contacts=config.ignored_contact_geom_ids[0],
                                         is_simplified=config.is_simplified, simplified_duration=config.simplified_duration, allow_approximate=config.allow_approximate, range_=config.range)
            self._simple_planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.simple_planner_type,
                                                passive_joint_idx=config.passive_joint_idx,
                                                ignored_contacts=config.ignored_contact_geom_ids[0], goal_bias=1.0, allow_approximate=False, is_simplified=config.simple_planner_simplified, simplified_duration=config.simple_planner_simplified_duration, range_=config.simple_planner_range)
            self._ac_rl_minimum = config.ac_rl_minimum
            self._ac_rl_maximum = config.ac_rl_maximum


    def _log_creation(self):
        if self._config.is_chef:
            logger.info('creating a TD3 agent')
            for i, _actor in enumerate(self._actors):
                logger.info('skill #{} has %d parameters'.format(i + 1), count_parameters(_actor))
            logger.info('the critic1 has %d parameters', count_parameters(self._critic1))
            logger.info('the critic2 has %d parameters', count_parameters(self._critic2))


    def _build_actor(self, actor):
        self._actors = [actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)]

    def _build_target_actor(self, actor):
        self._target_actors = [actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)]

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            'actor_state_dict': [_actor.state_dict() for _actor in self._actors],
            'critic1_state_dict': self._critic1.state_dict(),
            'critic2_state_dict': self._critic2.state_dict(),
            'actor_optim_state_dict': [_actor_optim.state_dict() for _actor_optim in self._actor_optims],
            'critic1_optim_state_dict': self._critic1_optim.state_dict(),
            'critic2_optim_state_dict': self._critic2_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        for _actor, actor_ckpt in zip(self._actors, ckpt['actor_state_dict']):
            _actor.load_state_dict(actor_ckpt)
        for _actor, _target_actor in zip(self._actors, self._target_actors):
            _target_actor.load_state_dict(_actor.state_dict())
        self._critic1.load_state_dict(ckpt['critic1_state_dict'])
        self._critic2.load_state_dict(ckpt['critic2_state_dict'])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        for _actor_optim, actor_optim_ckpt in zip(self._actor_optims, ckpt['actor_optim_state_dict']):
            _actor_optim.load_state_dict(actor_optim_ckpt)
        self._critic1_optim.load_state_dict(ckpt['critic1_optim_state_dict'])
        self._critic2_optim.load_state_dict(ckpt['critic2_optim_state_dict'])
        for _actor_optim in self._actor_optims:
            optimizer_cuda(_actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        for _actor, _target_actor in zip(self._actors, self._target_actors):
            _actor.to(device)
            _target_actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def sync_networks(self):
        for _actor in self._actors:
            sync_networks(_actor)
        sync_networks(self._critic1)
        sync_networks(self._critic2)

    def valid_action(self, ac):
        return np.all(ac['default'] >= -1.0) and np.all(ac['default'] <= 1.0)

    def is_planner_ac(self, ac):
        if np.any(ac['default'][:len(self._ref_joint_pos_indexes)] < self._ac_rl_minimum) or np.any(ac['default'][:len(self._ref_joint_pos_indexes)] > self._ac_rl_maximum):
            return True
        return False

    def isValidState(self, state):
        return self._planner.isValidState(state)

    def convert2planner_displacement(self, ac, ac_scale):
        ac_space_type = self._config.ac_space_type
        action_range = self._config.action_range
        ac_rl_maximum = self._config.ac_rl_maximum
        if ac_space_type == 'normal':
            return ac * action_range
        elif ac_space_type == 'pairwise':
            return np.where(np.abs(ac) < ac_rl_maximum, ac/ac_rl_maximum, np.sign(ac)*(ac_scale+(np.abs(ac)-ac_rl_maximum)*((action_range-ac_scale)/(action_range-ac_rl_maximum))))
        else:
            raise NotImplementedError

    def plan(self, curr_qpos, target_qpos, ac_scale=None, meta_ac=None, ob=None, is_train=True, random_exploration=False, ref_joint_pos_indexes=None):

        curr_qpos = self.clip_qpos(curr_qpos)
        interpolation = True
        if self._config.interpolate_type == 'planner':
            traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
        else:
            traj, success, valid, exact = self.simple_interpolate(curr_qpos, target_qpos, ac_scale)
        if not success:
            if self._config.allow_approximate:
                if self._config.allow_invalid:
                    traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos)
                    interpolation = False
                else:
                    if not exact:
                        traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos)
                        interpolation = False
            else:
                if not exact:
                    traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos, self._config.timelimit)
                    interpolation = False
                    if self._config.use_interpolation and success:
                        new_traj = []
                        start = curr_qpos
                        for i in range(len(traj)):
                            diff = traj[i] - start
                            if np.any(diff[:len(self._ref_joint_pos_indexes)] < -ac_scale) or np.any(diff[:len(self._ref_joint_pos_indexes)] > ac_scale):
                                if self._config.interpolate_type == 'planner':
                                    inner_traj, inner_success, inner_valid, inner_exact = self._simple_planner.plan(start, traj[i], self._config.simple_planner_timelimit)
                                    if inner_success:
                                        new_traj.extend(inner_traj)
                                else:
                                    inner_traj, _, _, _ = self.simple_interpolate(start, traj[i], ac_scale, use_planner=True)
                                    new_traj.extend(inner_traj)
                            else:
                                new_traj.append(traj[i])
                            start = traj[i]
                        traj = np.array(new_traj)

        return traj, success, interpolation, valid, exact

    def interpolate(self, curr_qpos, target_qpos):
        traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
        return traj, success, interpolation, valid, exact

    def clip_qpos(self, curr_qpos):
        tmp_pos = curr_qpos.copy()
        if np.any(curr_qpos[self._is_jnt_limited[self._jnt_indices]] < self._jnt_minimum[self._jnt_indices][self._is_jnt_limited[self._jnt_indices]]) or \
                np.any(curr_qpos[self._is_jnt_limited[self._jnt_indices]] > self._jnt_maximum[self._jnt_indices][self._is_jnt_limited[self._jnt_indices]]):
            new_curr_qpos = np.clip(curr_qpos.copy(), self._jnt_minimum[self._jnt_indices]+self._config.joint_margin, self._jnt_maximum[self._jnt_indices]-self._config.joint_margin)
            new_curr_qpos[np.invert(self._is_jnt_limited[self._jnt_indices])] = tmp_pos[np.invert(self._is_jnt_limited[self._jnt_indices])]
            curr_qpos = new_curr_qpos
        return curr_qpos

    def simple_interpolate(self, curr_qpos, target_qpos, ac_scale, use_planner=False):
        success = True
        exact = True
        curr_qpos = self.clip_qpos(curr_qpos)

        traj = []
        min_action = self._ac_space['default'].low[0] * ac_scale * 0.8
        max_action = self._ac_space['default'].high[0] * ac_scale * 0.8
        assert max_action > min_action, "action space box is ill defined"
        assert max_action > 0 and min_action < 0, "action space MAY be ill defined. Check this assertion"

        diff = target_qpos[:len(self._ref_joint_pos_indexes)] - curr_qpos[:len(self._ref_joint_pos_indexes)]
        out_of_bounds = np.where((diff > max_action) | (diff < min_action))[0]
        out_diff = diff[out_of_bounds]


        scales = np.where(out_diff > max_action, out_diff/max_action, out_diff/min_action)
        if len(scales) == 0:
            scaling_factor = 1.
        else:
            scaling_factor = max(max(scales), 1.)
        scaled_ac = diff[:len(self._ref_joint_pos_indexes)]

        valid = True
        interp_qpos = curr_qpos.copy()
        for i in range(int(scaling_factor)):
            interp_qpos[:len(self._ref_joint_pos_indexes)] += scaled_ac
            if not self._planner.isValidState(interp_qpos):
                valid = False
                break
            traj.append(interp_qpos.copy())

        if not valid and use_planner:
            traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
            if not success:
                traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos, self._config.timelimit)
                if not success:
                    traj = [target_qpos]
        else:
            if not valid:
                success = False
                exact = False
            traj.append(target_qpos)

        return np.array(traj), success, valid, exact


    def train(self):
        config = self._config
        for i in range(config.num_batches):
            transitions = self._buffer.sample(config.batch_size)
            train_info = self._update_network(transitions, step=i)
            for _actor, _target_actor in zip(self._actors, self._target_actors):
                self._soft_update_target_network(_target_actor, _actor, self._config.polyak)
            self._soft_update_target_network(self._critic1_target, self._critic1, self._config.polyak)
            self._soft_update_target_network(self._critic2_target, self._critic2, self._config.polyak)

        train_info.update({
            'actor_grad_norm': np.mean([compute_gradient_norm(_actor) for _actor in self._actors]),
            'actor_weight_norm': np.mean([compute_weight_norm(_actor) for _actor in self._actors]),
            'critic1_grad_norm': compute_gradient_norm(self._critic1),
            'critic2_grad_norm': compute_gradient_norm(self._critic2),
            'critic1_weight_norm': compute_weight_norm(self._critic1),
            'critic2_weight_norm': compute_weight_norm(self._critic2),
        })

        return train_info

    def act_log(self, ob, meta_ac=None):
        if meta_ac:
            raise NotImplementedError
        return self._actors[0].act_log(ob)

    def act(self, ob, is_train=True, return_stds=False):
        ob = to_tensor(ob, self._config.device)
        if return_stds:
            ac, activation, stds = self._actors[0].act(ob, is_train=is_train, return_stds=return_stds)
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    ac[k] += np.random.normal(0, 0.1, size=len(ac[k]))
                    ac[k] = np.clip(ac[k], self._config.action_min, self._config.action_max)
            return ac, activation, stds
        else:
            ac, activation = self._actors[0].act(ob, is_train=is_train, return_stds=return_stds)
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    ac[k] += np.random.normal(0, 0.1, size=len(ac[k]))
                    ac[k] = np.clip(ac[k], self._config.action_min, self._config.action_max)
            return ac, activation

    def target_act(self, ob, is_train=True):
        if self._config.policy == 'mlp':
            ob = self.normalize(ob)
        if hasattr(self, '_actor'):
            ac, activation = self._target_actors.act(ob, is_train=is_train)
        else:
            ac, activation = self._target_actors[0].act(ob, is_train=is_train)
        return ac, activation

    def target_act_log(self, ob, meta_ac=None):
        if meta_ac:
            raise NotImplementedError
        return self._target_actors[0].act_log(ob)

    def _update_network(self, transitions, step=0):
        config = self._config
        info = {}

        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        if config.hrl:
            meta_ac = _to_tensor(transitions['meta_ac'])
        else:
            meta_ac = None

        if 'intra_steps' in transitions.keys() and self._config.use_smdp_update:
            intra_steps = _to_tensor(transitions['intra_steps'])

        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        ## Actor loss
        actions_real, _ = self.act_log(o, meta_ac)
        actor_loss = -self._critic1(o, actions_real).mean()
        info['actor_loss'] = actor_loss.cpu().item()

        ## Critic loss
        with torch.no_grad():
            actions_next, _ = self.target_act_log(o_next, meta_ac)
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    actions_next[k] += torch.randn_like(actions_next[k]) * self._config.policy_noise
                    actions_next[k] = torch.clamp(actions_next[k], self._config.action_min, self._config.action_max)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2)
            target_q_value = rew + (1.-done) * config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()

        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)

        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real1_q'] = real_q_value1.min().cpu().item()
        info['min_real2_q'] = real_q_value2.min().cpu().item()
        info['real1_q'] = real_q_value1.mean().cpu().item()
        info['rea2_q'] = real_q_value2.mean().cpu().item()
        info['critic1_loss'] = critic1_loss.cpu().item()
        info['critic2_loss'] = critic2_loss.cpu().item()


        # update the critics
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic2)
        self._critic2_optim.step()

        # update the actor
        if step % self._config.actor_update_freq == 0:
            for _actor_optim in self._actor_optims:
                _actor_optim.zero_grad()
            actor_loss.backward()
            for i, _actor in enumerate(self._actors):
                sync_grads(_actor)
                self._actor_optims[i].step()

        return mpi_average(info)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
