import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.base_agent import BaseAgent
from rl.ppo_agent import PPOAgent
from rl.normalizer import Normalizer
from rl.dataset import LowLevelPPOReplayBuffer, RandomSampler
from rl.mp_agent import MpAgent
from util.mpi import mpi_average
from util.logger import logger
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, \
    obs2tensor, to_tensor, sync_avg_grads
from util.gym import action_size, observation_size
from env.action_spec import ActionSpec

from gym import spaces

from util.logger import logger

class LowLevelPPOSingleCriticAgent(BaseAgent):
    ''' Low level agent that includes skill sets for each agent, their
        execution procedure given observation and skill selections from
        meta-policy, and their training (for single-skill-per-agent cases
        only).
    '''

    def __init__(self, config, ob_space, ac_space, subgoal_space, actor, critic, non_limited_idx=None):
        self._non_limited_idx = non_limited_idx
        self._subgoal_space = subgoal_space
        self._ac_space = ac_space
        super().__init__(config, ob_space)

        self._actors = []
        self._old_actors = []
        for skill in config.primitive_skills:
            if 'mp' in skill:
                self._actors.append(actor(config, ob_space, subgoal_space, config.tanh_policy, activation='tanh'))
                self._old_actors.append(actor(config, ob_space, subgoal_space, config.tanh_policy, activation='tanh'))
            else:
                self._actors.append(actor(config, ob_space, ac_space, config.tanh_policy, activation='tanh'))
                self._old_actors.append(actor(config, ob_space, ac_space, config.tanh_policy, activation='tanh'))

        self._critic = critic(config, ob_space, activation='tanh')
        self._network_cuda(config.device)

        self._actor_optims = [optim.Adam(_actor.parameters(), lr=config.lr_actor) for _actor in self._actors]
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.lr_critic)

        self._build_planner()
        sampler = RandomSampler()
        self._buffer = LowLevelPPOReplayBuffer(['ob', 'ac', 'meta_ac', 'done', 'rew', 'ret', 'adv', 'ac_before_activation', 'vpred'],
                                    config.buffer_size,
                                    len(config.primitive_skills),
                                    sampler.sample_func)
        if config.is_chef:
            logger.info('Creating a PPO agent')
            for _actor in self._actors:
                logger.info('The actor has %d parameters', count_parameters(_actor))

    def _build_planner(self):
        config = self._config
        self._planners = []

        # Change here !!!!!!
        if config.primitive_skills:
            skills = config.primitive_skills
        else:
            skills = ['primitive']

        self._skills = skills
        planner_i = 0

        for skill in skills:
            if 'mp' in skill:
                ignored_contacts = config.ignored_contact_geom_ids[planner_i]
                passive_joint_idx = config.passive_joint_idx
                planner = MpAgent(config, self._ac_space, self._non_limited_idx, passive_joint_idx=passive_joint_idx, ignored_contacts=ignored_contacts)
                self._planners.append(planner)
                planner_i += 1
            else:
                self._planners.append(None)

    def plan(self, curr_qpos, target_qpos=None, meta_ac=None, ob=None, is_train=True, random_exploration=False, ref_joint_pos_indexes=None):
        assert len(self._planners) != 0, "No planner exists"

        if target_qpos is None:
            assert ob is not None and meta_ac is not None, "Invalid arguments"

            skill_idx = int(meta_ac['default'][0])
            assert self._planners[skill_idx] is not None

            assert "mp" in self.return_skill_type(meta_ac), "Skill is expected to be motion planner"
            if random_exploration:
                ac = self._ac_space.sample()
            else:
                ac, activation = self._actors[skill_idx].act(ob, is_train)
            target_qpos = curr_qpos.copy()
            target_qpos[ref_joint_pos_indexes] += ac['default'][:len(ref_joint_pos_indexes)]
            traj, success = self._planners[skill_idx].plan(curr_qpos, target_qpos)
            return traj, success, target_qpos, ac, activation
        else:
            traj, success = self._planners[0].plan(curr_qpos, target_qpos)
            return traj, success

    def act(self, ob, meta_ac, is_train=True, return_stds=False):
        if self._config.hrl:
            skill_idx = int(meta_ac['default'][0])
            if self._config.meta_update_target == 'HL':
                if return_stds:
                    ac, activation, stds = self._actors[skill_idx].act(ob, False, return_stds=return_stds)
                else:
                    ac, activation = self._actors[skill_idx].act(ob, False, return_stds=return_stds)
            else:
                if return_stds:
                    ac, activation, stds = self._actors[skill_idx].act(ob, is_train, return_stds=return_stds)
                else:
                    ac, activation = self._actors[skill_idx].act(ob, is_train, return_stds=return_stds)

        if return_stds:
            return ac, activation, stds
        else:
            return ac, activation

    def get_value(self, ob, meta_ac):
        ob = obs2tensor(ob, self._config.device)
        return self._critic(ob).detach().cpu().numpy()[:, 0]

    def return_skill_type(self, meta_ac):
        skill_idx = int(meta_ac['default'][0])
        return self._skills[skill_idx]

    def act_log(self, ob, meta_ac=None):
        ''' Note: only usable for SAC agents '''
        skill_idx = int(meta_ac['default'][0])
        return self._actors[skill_idx].act_log(ob)

    def store_episode(self, rollouts):
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def _compute_gae(self, rollouts):
        T = len(rollouts['done'])
        ob = rollouts['ob']
        # ob = self.normalize(ob)
        ob = obs2tensor(ob, self._config.device)
        vpred = self._critic(ob).detach().cpu().numpy()[:,0]
        assert len(vpred) == T + 1

        done = rollouts['done']
        rew = rollouts['rew']
        adv = np.empty((T, ) , 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = rew[t] + self._config.discount_factor * vpred[t+1] * nonterminal - vpred[t]
            adv[t] = lastgaelam = delta + self._config.discount_factor * self._config.gae_lambda * nonterminal * lastgaelam

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        rollouts['adv'] = ((adv - adv.mean()) / (adv.std()+1e-5)).tolist()
        rollouts['ret'] = ret.tolist()


    def train(self):
        train_info = {}
        for i in range(len(self._actors)):
            self._soft_update_target_network(self._old_actors[i], self._actors[i], 0.0)
        for skill_idx in range(len(self._config.primitive_skills)):
            sample_size = len(self._buffer._buffers[skill_idx]['ac'])
            iters = max(int(sample_size // self._config.batch_size), 1)
            for _ in range(iters*self._config.num_batches):
                if self._buffer._current_size[skill_idx] > 0:
                    transitions = self._buffer.sample(self._config.batch_size, skill_idx)
                else:
                    transitions = self._buffer.create_empty_transition()

                info = self._update_network(transitions, skill_idx)
                train_info.update(info)
                train_info.update({
                    'actor_grad_norm': compute_gradient_norm(self._actors[skill_idx]),
                    'actor_weight_norm': compute_weight_norm(self._actors[skill_idx]),
                    'critic_grad_norm': compute_gradient_norm(self._critic),
                    'critic_weight_norm': compute_weight_norm(self._critic),
                })

        self._buffer.clear()

        return train_info

    def state_dict(self):
        return {
            'actor_state_dict': [_actor.state_dict() for _actor in self._actors],
            'critic_state_dict': self._critic.state_dict(),
            'actor_optim_state_dict': [_actor_optim.state_dict() for _actor_optim in self._actor_optims],
            'critic_optim_state_dict': self._critic_optim.state_dict(),
            'ob_norm_state_dict': self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        for _actor, _actor_ckpt in zip(self._actrs, ckpt['actor_state_dict']):
            _actor.load_state_dict(_actor_ckpt)
        self._critic.load_state_dict(ckpt['critic_state_dict'])
        self._ob_norm.load_state_dict(ckpt['ob_norm_state_dict'])
        self._network_cuda(self._config.device)

        for _actor_optim, _actor_optim_ckpt in zip(self._actor_optims, ckpt['actor_optim_state_dict']):
            _actor_optim.load_state_dict(_actor_optim_ckpt)
        self._critic_optim.load_state_dict(ckpt['critic_optim_state_dict'])
        for _actor_optim in self._actor_optims:
            optimizer_cuda(_actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        for _actor, _old_actor in zip(self._actors, self._old_actors):
            _actor.to(device)
            _old_actor.to(device)
        self._critic.to(device)

    def sync_networks(self):
        for _actor in self._actors:
            sync_networks(_actor)
        sync_networks(self._critic)

    def _update_network(self, transitions, skill_idx):
        info = {}
        postfix = '_' + self._config.primitive_skills[skill_idx]

        # pre-process observations
        o = transitions['ob']
        # o = self.normalize(o)
        if len(o) == 0:
            self._actor_optims[skill_idx].zero_grad()
            sync_avg_grads(self._actors[skill_idx])
            self._actor_optims[skill_idx].step()

            # update the critic
            self._critic_optim.zero_grad()
            sync_avg_grads(self._critic)
            self._critic_optim.step()

            info['value_target{}'.format(postfix)] = 0.
            info['value_predicted{}'.format(postfix)] = 0.
            info['value_loss{}'.format(postfix)] = 0.
            info['actor_loss{}'.format(postfix)] = 0.
            info['entropy_loss{}'.format(postfix)] = 0.
            return mpi_average(info)

        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions['ac'])
        a_z = _to_tensor(transitions['ac_before_activation'])
        ret = _to_tensor(transitions['ret']).reshape(bs, 1)
        adv = _to_tensor(transitions['adv']).reshape(bs, 1)

        log_pi, ent = self._actors[skill_idx].act_log(o, a_z)
        old_log_pi, _ = self._old_actors[skill_idx].act_log(o, a_z)
        if old_log_pi.min() < -100:
            import ipdb; ipdb.set_trace()

        # the actor loss
        entropy_loss = self._config.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(log_pi - old_log_pi.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self._config.clip_param,
                            1.0 + self._config.clip_param) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        if not np.isfinite(ratio.cpu().detach()).all() or not np.isfinite(adv.cpu().detach()).all():
            import ipdb; ipdb.set_trace()
        info['entropy_loss{}'.format(postfix)] = entropy_loss.cpu().item()
        info['actor_loss{}'.format(postfix)] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # the q loss
        value_pred = self._critic(o)
        value_loss = self._config.value_loss_coeff * (ret - value_pred).pow(2).mean()

        info['value_target{}'.format(postfix)] = ret.mean().cpu().item()
        info['value_predicted{}'.format(postfix)] = value_pred.mean().cpu().item()
        info['value_loss{}'.format(postfix)] = value_loss.cpu().item()

        # update the actor
        self._actor_optims[skill_idx].zero_grad()
        actor_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._actors[skill_idx].parameters(), self._config.max_grad_norm)
        sync_avg_grads(self._actors[skill_idx])
        self._actor_optims[skill_idx].step()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._config.max_grad_norm)
        sync_avg_grads(self._critic)
        self._critic_optim.step()

        # include info from policy
        # info.update(self._actor.info)

        return mpi_average(info)
