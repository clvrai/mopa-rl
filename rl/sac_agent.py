#] SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

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

class SACAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space,
                 actor, critic, non_limited_idx=None, ref_joint_pos_indexes=None, joint_space=None, is_jnt_limited=None, jnt_indices=None):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._jnt_indices = jnt_indices
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._log_alpha = torch.tensor(np.log(config.alpha), requires_grad=True, device=config.device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.lr_actor)
        self._joint_space = joint_space
        self._is_jnt_limited = is_jnt_limited
        if joint_space is not None:
            self._jnt_minimum = joint_space['default'].low
            self._jnt_maximum = joint_space['default'].high

        # build up networks
        self._build_actor(actor)
        self._build_critic(critic)
        self._network_cuda(config.device)

        self._target_entropy = -action_size(self._actor._ac_space)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.lr_actor)
        self._critic1_optim = optim.Adam(self._critic1.parameters(), lr=config.lr_critic)
        self._critic2_optim = optim.Adam(self._critic2.parameters(), lr=config.lr_critic)

        sampler = RandomSampler()
        buffer_keys = ['ob', 'ac', 'meta_ac', 'done', 'rew']
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
                                                ignored_contacts=config.ignored_contact_geom_ids, goal_bias=1.0, allow_approximate=False,
                                                is_simplified=config.simple_planner_simplified, simplified_duration=config.simple_planner_simplified_duration, range_=config.simple_planner_range)
            self._omega = config.omega


    def _log_creation(self):
        if self._config.is_chef:
            logger.info('creating a sac agent')
            logger.info('the Actor has %d parameters', count_parameters(self._actor))
            logger.info('the critic1 has %d parameters', count_parameters(self._critic1))
            logger.info('the critic2 has %d parameters', count_parameters(self._critic2))

    def _build_actor(self, actor):
        self._actor = actor(self._config, self._ob_space, self._ac_space,
                               self._config.tanh_policy)

    def _build_critic(self, critic):
        config = self._config
        self._critic1 = critic(config, self._ob_space, self._ac_space)
        self._critic2 = critic(config, self._ob_space, self._ac_space)

        # build up target networks
        self._critic1_target = critic(config, self._ob_space, self._ac_space)
        self._critic2_target = critic(config, self._ob_space, self._ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())

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
            'log_alpha': self._log_alpha.cpu().detach().numpy(),
            'actor_state_dict': self._actor.state_dict(),
            'critic1_state_dict': self._critic1.state_dict(),
            'critic2_state_dict': self._critic2.state_dict(),
            'alpha_optim_state_dict': self._alpha_optim.state_dict(),
            'actor_optim_state_dict': self._actor_optim.state_dict(),
            'critic1_optim_state_dict': self._critic1_optim.state_dict(),
            'critic2_optim_state_dict': self._critic2_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._log_alpha.data = torch.tensor(ckpt['log_alpha'], requires_grad=True,
                                            device=self._config.device)
        self._actor.load_state_dict(ckpt['actor_state_dict'])
        self._critic1.load_state_dict(ckpt['critic1_state_dict'])
        self._critic2.load_state_dict(ckpt['critic2_state_dict'])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt['alpha_optim_state_dict'])
        self._actor_optim.load_state_dict(ckpt['actor_optim_state_dict'])
        self._critic1_optim.load_state_dict(ckpt['critic1_optim_state_dict'])
        self._critic2_optim.load_state_dict(ckpt['critic2_optim_state_dict'])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
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
        for i in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(self._critic1_target, self._critic1, self._config.polyak)
            self._soft_update_target_network(self._critic2_target, self._critic2, self._config.polyak)
        return train_info

    def act_log(self, ob, meta_ac=None):
        #assert meta_ac is None, "vanilla SAC agent doesn't support meta action input"
        if meta_ac:
            raise NotImplementedError()
        return self._actor.act_log(ob)

    def _update_network(self, transitions, step=0):
        info = {}

        # pre-process observations
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])

        if 'intra_steps' in transitions.keys() and self._config.use_smdp_update:
            intra_steps = _to_tensor(transitions['intra_steps'])

        if self._config.hrl:
            meta_ac = _to_tensor(transitions['meta_ac'])
        else:
            meta_ac = None
        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        # update alpha
        if self._config.log_indiv_entropy:
            actions_real, log_pi, entropies = self.act_log(o, meta_ac=meta_ac)
            for key in entropies.keys():
                info['entropy_'  + key] = entropies[key].mean().cpu().item()
        else:
            actions_real, log_pi = self.act_log(o, meta_ac=meta_ac)
        alpha_loss = -(self._log_alpha.exp() * (log_pi + self._target_entropy).detach()).mean()

        if self._config.use_automatic_entropy_tuning:
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            alpha = self._log_alpha.exp()
            info['alpha_loss'] = alpha_loss.cpu().item()
            info['entropy_alpha'] = alpha.cpu().item()
            alpha = self._log_alpha.exp()
        else:
            alpha = torch.ones(1, device=self._config.device) * self._config.alpha


        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(self._critic1(o, actions_real),
                                self._critic2(o, actions_real)).mean()
        info['log_pi'] = log_pi.mean().cpu().item()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # calculate the target Q value function
        with torch.no_grad():
            if self._config.log_indiv_entropy:
                actions_next, log_pi_next, _ = self.act_log(o_next, meta_ac=meta_ac)
            else:
                actions_next, log_pi_next = self.act_log(o_next, meta_ac=meta_ac)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            if meta_ac is None:
                q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            if self._config.use_smdp_update:
                target_q_value = self._config.reward_scale * rew + \
                    (1 - done) * (self._config.discount_factor ** (intra_steps+1)) * q_next_value
            else:
                target_q_value = self._config.reward_scale * rew + \
                    (1 - done) * self._config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()
            ## clip the q value
            # clip_return = 1 / (1 - self._config.discount_factor)
            # target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                ac[k] = F.one_hot(ac[k].long(), action_size(self._ac_space[k])).float().squeeze(1)
        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real1_q'] = real_q_value1.min().cpu().item()
        info['min_real2_q'] = real_q_value2.min().cpu().item()
        info['real1_q'] = real_q_value1.mean().cpu().item()
        info['real2_q'] = real_q_value2.mean().cpu().item()
        info['critic1_loss'] = critic1_loss.cpu().item()
        info['critic2_loss'] = critic2_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._config.is_mpi:
            sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        if self._config.is_mpi:
            sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        if self._config.is_mpi:
            sync_grads(self._critic2)
        self._critic2_optim.step()

        if self._config.is_mpi:
            return mpi_average(info)
        else:
            return info
