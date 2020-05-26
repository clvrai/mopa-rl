# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dataset import ReplayBuffer, RandomSampler, LowLevelReplayBuffer
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
                 actor, critic, non_limited_idx=None, ref_joint_pos_indexes=None):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._ref_joint_pos_indexes = ref_joint_pos_indexes
        self._log_alpha = [torch.zeros(1, requires_grad=True, device=config.device)]
        self._alpha_optim = [optim.Adam([self._log_alpha[0]], lr=config.lr_actor)]

        # build up networks
        self._build_actor(actor)
        self._build_critic(critic)
        self._network_cuda(config.device)

        self._target_entropy = [-action_size(_actor._ac_space) for _actor in self._actors]

        self._actor_optims = [optim.Adam(_actor.parameters(), lr=config.lr_actor) for _actor in self._actors]
        self._critic1_optims = [optim.Adam(_critic.parameters(), lr=config.lr_actor) for _critic in self._critics1]
        self._critic2_optims = [optim.Adam(_critic.parameters(), lr=config.lr_actor) for _critic in self._critics2]

        sampler = RandomSampler()
        buffer_keys = ['ob', 'ac', 'meta_ac', 'done', 'rew']
        if config.hrl:
            self._buffer = LowLevelReplayBuffer(buffer_keys,
                                                config.buffer_size,
                                                len(config.primitive_skills),
                                                sampler.sample_func)
        else:
            self._buffer = ReplayBuffer(buffer_keys,
                                        config.buffer_size,
                                        sampler.sample_func)

        self._log_creation()

        self._planner = None
        self._is_planner_initialized = False
        if config.planner_integration:
            self._planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.planner_type,
                                         passive_joint_idx=config.passive_joint_idx, ignored_contacts=config.ignored_contact_geom_ids[0],
                                         is_simplified=config.is_simplified, simplified_duration=config.simplified_duration, allow_approximate=config.allow_approximate)
            self._simple_planner = PlannerAgent(config, ac_space, non_limited_idx, planner_type=config.simple_planner_type,
                                                passive_joint_idx=config.passive_joint_idx,
                                                ignored_contacts=config.ignored_contact_geom_ids[0], goal_bias=1.0, allow_approximate=False, is_simplified=config.simple_planner_simplified, simplified_duration=config.simple_planner_simplified_duration)
            self._ac_rl_minimum = config.ac_rl_minimum
            self._ac_rl_maximum = config.ac_rl_maximum


    def _log_creation(self):
        if self._config.is_chef:
            logger.info('creating a sac agent')
            for i, _actor in enumerate(self._actors):
                logger.info('skill #{} has %d parameters'.format(i + 1), count_parameters(_actor))
            logger.info('the critic1 has %d parameters', count_parameters(self._critics1[0]))
            logger.info('the critic2 has %d parameters', count_parameters(self._critics2[0]))

    def _build_actor(self, actor):
        self._actors = [actor(self._config, self._ob_space, self._ac_space,
                               self._config.tanh_policy)] # num_body_parts, num_skills

    def _build_critic(self, critic):
        config = self._config
        self._critics1 = [critic(config, self._ob_space, self._ac_space)]
        self._critics2 = [critic(config, self._ob_space, self._ac_space)]

        # build up target networks
        self._critic1_targets = [critic(config, self._ob_space, self._ac_space)]
        self._critic2_targets = [critic(config, self._ob_space, self._ac_space)]
        self._critic1_targets[0].load_state_dict(self._critics1[0].state_dict())
        self._critic2_targets[0].load_state_dict(self._critics2[0].state_dict())

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def valid_action(self, ac):
        return np.all(ac['default'] >= -1.0) and np.all(ac['default'] <= 1.0)


    def is_planner_ac(self, ac):
        if np.any(ac['default'][:len(self._ref_joint_pos_indexes)] < self._ac_rl_minimum) or np.any(ac['default'][:len(self._ref_joint_pos_indexes)] > self._ac_rl_maximum):
            return True
        return False

    def plan(self, curr_qpos, target_qpos, ac_scale=None, meta_ac=None, ob=None, is_train=True, random_exploration=False, ref_joint_pos_indexes=None):
        if self._config.use_double_planner:
            interpolation = True
            traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
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
                        traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos)
                        interpolation = False
        else:
            interpolation = False
            traj, success, valid, exact = self._planner.plan(curr_qpos, target_qpos)
            if self._config.planner_type == 'prm_star' and success:
                new_traj = []
                start = curr_qpos
                for i in range(len(traj)):
                    diff = traj[i] - start
                    if np.any(diff[:len(self._ref_joint_pos_indexes)] < -ac_scale) or np.any(diff[:len(self._ref_joint_pos_indexes)] > ac_scale):
                        inner_traj, inner_success, inner_valid, inner_exact = self._simple_planner.plan(start, traj[i], self._config.simple_planner_timelimit)
                        if inner_success:
                            new_traj.extend(inner_traj)
                    else:
                        new_traj.append(traj[i])
                    start = traj[i]
                traj = np.array(new_traj)

        return traj, success, interpolation, valid, exact

    def interpolate(self, curr_qpos, target_qpos):
        traj, success, valid, exact = self._simple_planner.plan(curr_qpos, target_qpos, self._config.simple_planner_timelimit)
        return traj, success, interpolation, valid, exact

    def state_dict(self):
        return {
            'log_alpha': [_log_alpha.cpu().detach().numpy() for _log_alpha in self._log_alpha],
            'actor_state_dict': [_actor.state_dict() for _actor in self._actors],
            'critic1_state_dict': [_critic1.state_dict() for _critic1 in self._critics1],
            'critic2_state_dict': [_critic2.state_dict() for _critic2 in self._critics2],
            'alpha_optim_state_dict': [_alpha_optim.state_dict() for _alpha_optim in self._alpha_optim],
            'actor_optim_state_dict': [_actor_optim.state_dict() for _actor_optim in self._actor_optims],
            'critic1_optim_state_dict': [_critic1_optim.state_dict() for _critic1_optim in self._critic1_optims],
            'critic2_optim_state_dict': [_critic2_optim.state_dict() for _critic2_optim in self._critic2_optims],
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        for _log_alpha, _log_alpha_ckpt in zip(self._log_alpha, ckpt['log_alpha']):
            _log_alpha.data = torch.tensor(_log_alpha_ckpt, requires_grad=True,
                                                device=self._config.device)
        for _actor, actor_ckpt in zip(self._actors, ckpt['actor_state_dict']):
            _actor.load_state_dict(actor_ckpt)
        for _critic1, critic_ckpt in zip(self._critics1, ckpt['critic1_state_dict']):
            _critic1.load_state_dict(critic_ckpt)
        for _critic2, critic_ckpt in zip(self._critics2, ckpt['critic2_state_dict']):
            _critic2.load_state_dict(critic_ckpt)

        for _critic1, _critic1_target in zip(self._critics1, self._critic1_targets):
            _critic1_target.load_state_dict(_critic1.state_dict())
        for _critic2, _critic2_target in zip(self._critics2, self._critic2_targets):
            _critic2_target.load_state_dict(_critic2.state_dict())

        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        for _alpha_optim, _alpha_optim_ckpt in zip(self._alpha_optim, ckpt['alpha_optim_state_dict']):
            _alpha_optim.load_state_dict(_alpha_optim_ckpt)
        for _actor_optim, actor_optim_ckpt in zip(self._actor_optims, ckpt['actor_optim_state_dict']):
            _actor_optim.load_state_dict(actor_optim_ckpt)
        for _critic_optim, critic_optim_ckpt in zip(self._critic1_optims, ckpt['critic1_optim_state_dict']):
            _critic_optim.load_state_dict(critic_optim_ckpt)
        for _critic_optim, critic_optim_ckpt in zip(self._critic1_optims, ckpt['critic2_optim_state_dict']):
            _critic_optim.load_state_dict(critic_optim_ckpt)

        for _alpha_optim in self._alpha_optim:
            optimizer_cuda(_alpha_optim, self._config.device)
        for _actor_optim in self._actor_optims:
            optimizer_cuda(_actor_optim, self._config.device)
        for _critic_optim in self._critic1_optims:
            optimizer_cuda(_critic_optim, self._config.device)
        for _critic_optim in self._critic2_optims:
            optimizer_cuda(_critic_optim, self._config.device)

    def _network_cuda(self, device):
        for _actor in self._actors:
            _actor.to(device)
        for _critic in self._critics1:
            _critic.to(device)
        for _critic in self._critics2:
            _critic.to(device)
        for _critic_target in self._critic1_targets:
            _critic_target.to(device)
        for _critic_target in self._critic2_targets:
            _critic_target.to(device)

    def sync_networks(self):
        if self._config.is_mpi:
            for _actor in self._actors:
                sync_networks(_actor)
            for _critic in self._critics1:
                sync_networks(_critic)
            for _critic in self._critics2:
                sync_networks(_critic)

    def train(self):
        for i in range(self._config.num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            train_info = self._update_network(transitions, i)
            self._soft_update_target_network(self._critic1_targets[0], self._critics1[0], self._config.polyak)
            self._soft_update_target_network(self._critic2_targets[0], self._critics2[0], self._config.polyak)

        train_info.update({
            'actor_grad_norm': np.mean([compute_gradient_norm(_actor) for _actor in self._actors]),
            'actor_weight_norm': np.mean([compute_weight_norm(_actor) for _actor in self._actors]),
            'critic1_grad_norm': np.mean([compute_gradient_norm(_critic1) for _critic1 in self._critics1]),
            'critic2_grad_norm': np.mean([compute_gradient_norm(_critic2) for _critic2 in self._critics2]),
            'critic1_weight_norm': np.mean([compute_weight_norm(_critic1) for _critic1 in self._critics1]),
            'critic2_weight_norm': np.mean([compute_weight_norm(_critic2) for _critic2 in self._critics2]),
        })
        # print(train_info)

        return train_info

    def act_log(self, ob, meta_ac=None):
        #assert meta_ac is None, "vanilla SAC agent doesn't support meta action input"
        if meta_ac:
            raise NotImplementedError()
        return self._actors[0].act_log(ob)

    def _update_network(self, transitions, step=0):
        info = {}

        # pre-process observations
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        if self._config.hrl:
            meta_ac = _to_tensor(transitions['meta_ac'])
        else:
            meta_ac = None
        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        # update alpha
        actions_real, log_pi = self.act_log(o, meta_ac=meta_ac)
        if self._config.use_automatic_entropy_tuning:
            alpha = [_log_alpha.exp() for _log_alpha in self._log_alpha]
        else:
            alpha = [torch.ones(1, device=self._config.device) * self._config.alpha for _ in self._log_alpha]


        # the actor loss
        entropy_loss = (alpha[0] * log_pi).mean()
        actor_loss = -torch.min(self._critics1[0](o, actions_real),
                                self._critics2[0](o, actions_real)).mean()
        info['log_pi'] = log_pi.mean().cpu().item()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next, meta_ac=meta_ac)
            q_next_value1 = self._critic1_targets[0](o_next, actions_next)
            q_next_value2 = self._critic2_targets[0](o_next, actions_next)
            if meta_ac is None:
                q_next_value = torch.min(q_next_value1, q_next_value2) - alpha[0] * log_pi_next
            target_q_value = rew * self._config.reward_scale + \
                (1 - done) * self._config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()
            ## clip the q value
            # clip_return = 1 / (1 - self._config.discount_factor)
            # target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        for k, space in self._ac_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                ac[k] = F.one_hot(ac[k].long(), action_size(self._ac_space[k])).float().squeeze(1)
        real_q_value1 = self._critics1[0](o, ac)
        real_q_value2 = self._critics2[0](o, ac)
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
        for _actor_optim in self._actor_optims:
            _actor_optim.zero_grad()
        actor_loss.backward()
        for i, _actor in enumerate(self._actors):
            if self._config.is_mpi:
                sync_grads(_actor)
            self._actor_optims[i].step()

        # update the critic
        for _critic1_optim in self._critic1_optims:
            _critic1_optim.zero_grad()
        critic1_loss.backward()
        for i, _critic1 in enumerate(self._critics1):
            if self._config.is_mpi:
                sync_grads(_critic1)
            self._critic1_optims[i].step()

        for _critic2_optim in self._critic2_optims:
            _critic2_optim.zero_grad()
        critic2_loss.backward()
        for i, _critic2 in enumerate(self._critics2):
            if self._config.is_mpi:
                sync_grads(_critic2)
            self._critic2_optims[i].step()

        actions_real, log_pi = self.act_log(o, meta_ac=meta_ac)
        alpha_loss = -(self._log_alpha[0].exp() * (log_pi + self._target_entropy[0]).detach()).mean()

        if self._config.use_automatic_entropy_tuning:
            self._alpha_optim[0].zero_grad()
            alpha_loss.backward()
            self._alpha_optim[0].step()
            alpha = [_log_alpha.exp() for _log_alpha in self._log_alpha]
            info['alpha_loss'] = alpha_loss.cpu().item()
            info['entropy_alpha'] = alpha[0].cpu().item()

        if self._config.is_mpi:
            return mpi_average(info)
        else:
            return info
