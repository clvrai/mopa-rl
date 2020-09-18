from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dataset import ReplayBuffer, RandomSampler
from rl.base_agent import BaseAgent
from rl.planner_agent import PlannerAgent
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor
from util.gym import action_size, observation_size
from gym import spaces

class TD3Agent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic, non_limited_idx=None, ref_joint_pos_indexes=None, joint_space=None,
                 is_jnt_limited=None, jnt_indices=None):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        if joint_space is not None:
            self._jnt_minimum = joint_space['default'].low
            self._jnt_maximum = joint_space['default'].high

        self._log_alpha = [torch.zeros(1, requires_grad=True, device=config.device)]
        self._alpha_optim = [optim.Adam([self._log_alpha[0]], lr=config.lr_actor)]

        self._actor = actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)
        self._actor_target = actor(self._config, self._ob_space,
                              self._ac_space, self._config.tanh_policy, deterministic=True)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic1 = critic(config, ob_space, ac_space)
        self._critic2 = critic(config, ob_space, ac_space)
        self._critic1_target = critic(config, ob_space, ac_space)
        self._critic2_target = critic(config, ob_space, ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic1_optim = optim.Adam(self._critic1.parameters(), lr=config.lr_critic)
        self._critic2_optim = optim.Adam(self._critic2.parameters(), lr=config.lr_critic)

        self._update_steps = 0

        sampler = RandomSampler()
        buffer_keys = ['ob', 'ac', 'done', 'rew']
        if config.mopa or config.expand_ac_space:
            buffer_keys.append("intra_steps")
        self._buffer = ReplayBuffer(buffer_keys,
                                    config.buffer_size,
                                    sampler.sample_func)

        self._log_creation()

        self._planner = None
        self._is_planner_initialized = False
        if config.mopa:
            self._planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.planner_type,
                                         passive_joint_idx=config.passive_joint_idx, ignored_contacts=config.ignored_contact_geom_ids,
                                         is_simplified=config.is_simplified, simplified_duration=config.simplified_duration, allow_approximate=config.allow_approximate, range_=config.range)
            self._simple_planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.simple_planner_type,
                                                passive_joint_idx=config.passive_joint_idx,
                                                ignored_contacts=config.ignored_contact_geom_ids, goal_bias=1.0, allow_approximate=False, is_simplified=config.simple_planner_simplified, simplified_duration=config.simple_planner_simplified_duration, range_=config.simple_planner_range)
            self._omega = config.omega

    def _log_creation(self):
        logger.info('creating a TD3 agent')
        logger.info('the actor has %d parameters', count_parameters(self._actor))
        logger.info('the critic1 has %d parameters', count_parameters(self._critic1))
        logger.info('the critic2 has %d parameters', count_parameters(self._critic2))

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def valid_action(self, ac):
        return np.all(ac['default'] >= -1.0) and np.all(ac['default'] <= 1.0)

    def is_planner_ac(self, ac):
        if np.any(ac['default'][:len(self._ref_joint_pos_indexes)] < -self._omega) or np.any(ac['default'][:len(self._ref_joint_pos_indexes)] > self._omega):
            return True
        return False

    def isValidState(self, state):
        return self._planner.isValidState(state)

    def convert2planner_displacement(self, ac, ac_scale):
        ac_space_type = self._config.ac_space_type
        action_range = self._config.action_range
        if ac_space_type == 'normal':
            return ac * action_range
        elif ac_space_type == 'piecewise':
            return np.where(np.abs(ac) < self._omega, ac/(self._omega/ac_scale), np.sign(ac) * (ac_scale + (action_range-ac_scale)*((np.abs(ac)-self._omega)/(1-self._omega))))
        else:
            raise NotImplementedError

    def invert_displacement(self, displacement, ac_scale):
        ac_space_type = self._config.ac_space_type
        action_range = self._config.action_range
        if ac_space_type == 'normal':
            return displacement / action_range
        elif ac_space_type == 'piecewise':
            return np.where(np.abs(displacement)<ac_scale, displacement*(self._omega/ac_scale), np.sign(displacement) * ((np.abs(displacement) - ac_scale)/((action_range-ac_scale)/(1.0-ac_scale))/((1.0-ac_scale)/(1.0-self._omega))+self._omega))
        else:
            raise NotImplementedError

    # Calls motion planner to plan a path
    def plan(self, curr_qpos, target_qpos, ac_scale=None, meta_ac=None, ob=None, is_train=True, random_exploration=False, ref_joint_pos_indexes=None):

        curr_qpos = self.clip_qpos(curr_qpos)
        interpolation = True
        if self._config.interpolate_type == 'planner':
            traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
        else:
            traj, success, valid, exact = self.simple_interpolate(curr_qpos, target_qpos, ac_scale)
        if not success:
            if not exact or self._config.allow_approximate:
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

    # interpolation function
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
        scaled_ac = diff[:len(self._ref_joint_pos_indexes)] / scaling_factor

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
                    success = False
                    exact = False
        else:
            if not valid:
                success = False
                exact = False
            traj.append(target_qpos)

        return np.array(traj), success, valid, exact


    def state_dict(self):
        return {
            'actor_state_dict': self._actor.state_dict(),
            'critic1_state_dict': self._critic1.state_dict(),
            'critic2_state_dict': self._critic2.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic1_optim_state_dict': self._critic1_optim.state_dict(),
            'critic2_optim_state_dict': self._critic2_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._critic1.load_state_dict(ckpt['critic1_state_dict'])
        self._critic2.load_state_dict(ckpt['critic2_state_dict'])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic1_optim.load_state_dict(ckpt['critic1_optim_state_dict'])
        self._critic2_optim.load_state_dict(ckpt['critic2_optim_state_dict'])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._actor_target.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def sync_networks(self):
        if self._config.is_mpi:
            sync_networks(self._actor)
            sync_networks(self._critic1)
            sync_networks(self._critic2)

    def train(self):
        config = self._config
        for i in range(config.num_batches):
            transitions = self._buffer.sample(config.batch_size)
            train_info = self._update_network(transitions, step=i)
            if self._update_steps % self._config.actor_update_freq:
                self._soft_update_target_network(self._actor_target, self._actor, self._config.polyak)
                self._soft_update_target_network(self._critic1_target, self._critic1, self._config.polyak)
                self._soft_update_target_network(self._critic2_target, self._critic2, self._config.polyak)
        return train_info

    def act_log(self, ob, meta_ac=None):
        return self._actor.act_log(ob)

    def act(self, ob, is_train=True, return_stds=False):
        ob = to_tensor(ob, self._config.device)
        if return_stds:
            ac, activation, stds = self._actor.act(ob, is_train=is_train, return_stds=return_stds)
        else:
            ac, activation = self._actor.act(ob, is_train=is_train)
        if is_train:
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    ac[k] += np.random.normal(0, self._config.action_noise, size=len(ac[k]))
                    ac[k] = np.clip(ac[k], self._ac_space[k].low, self._ac_space[k].high)
        if return_stds:
            return ac, activation, stds
        else:
            return ac, activation

    def target_act(self, ob, is_train=True):
        ac, activation = self._actor_target.act(ob, is_train=is_train)
        return ac, activation

    def target_act_log(self, ob):
        return self._actor_target.act_log(ob)

    def _update_network(self, transitions, step=0):
        config = self._config
        info = {}

        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])

        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        ## Actor loss
        actions_real, _ = self.act_log(o)
        actor_loss = -self._critic1(o, actions_real).mean()
        info['actor_loss'] = actor_loss.cpu().item()

        ## Critic loss
        with torch.no_grad():
            actions_next, _ = self.target_act_log(o_next)
            for k, space in self._ac_space.spaces.items():
                if isinstance(space, spaces.Box):
                    epsilon = torch.randn_like(actions_next[k]) * self._config.target_noise
                    epsilon = torch.clamp(epsilon, -config.noise_clip, config.noise_clip)
                    actions_next[k] += epsilon
                    actions_next[k].clamp(-1., 1.)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2)
            target_q_value = rew * self._config.reward_scale + (1.-done) * config.discount_factor * q_next_value
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

        if self._update_steps % self._config.actor_update_freq == 0:
            # update the actor
            self._actor_optim.zero_grad()
            actor_loss.backward()
            self._actor_optim.step()

        # update the critics
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        self._critic2_optim.step()
        self._update_steps += 1

        return info
