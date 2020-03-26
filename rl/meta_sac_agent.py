import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.dataset import ReplayBuffer, RandomSampler
from rl.base_agent import BaseAgent
from rl.sac_agent import SACAgent
from rl.policies.mlp_actor_critic import MlpActor, MlpCritic
from rl.policies.cnn_actor_critic import CNNActor, CNNCritic
from util.logger import logger
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, \
    obs2tensor, to_tensor
from env.action_spec import ActionSpec
from util.gym import action_size
from gym import spaces
from rl.policies import get_actor_critic_by_name


class MetaSACAgent(SACAgent):
    def __init__(self, config, ob_space, joint_space=None, sampler=None):

        if not config.hrl:
            logger.warn('Creating a dummy meta SAC agent')
            return

        if config.primitive_skills:
            skills = config.primitive_skills
        else:
            skills = ['primitive']
        self.skills = skills


        if config.hrl:
            ac_space = spaces.Dict()
            ac_space.spaces['default'] = spaces.Discrete(len(skills))
            if config.hl_type == 'subgoal':
                if config.subgoal_type == 'joint':
                    ac_space.spaces['subgoal'] = spaces.Box(shape=(action_size(joint_space),), low=-1., high=1.)
                else:
                    ac_space.spaces['subgoal'] = spaces.Box(shape=(2,), low=-0.4, high=0.4)
            self.ac_space = ac_space

        # build up networks
        actor, critic = get_actor_critic_by_name(config.policy, config.use_ae)

        super().__init__(config, ob_space, ac_space, actor, critic)

        if sampler is None:
            sampler = RandomSampler()
        #buffer_keys = ['ob', 'ac', 'done', 'rew', 'ag', 'g']
        buffer_keys = ['ob', 'ac', 'done', 'rew']
        self._buffer = ReplayBuffer(buffer_keys,
                                    config.buffer_size,
                                    sampler.sample_func)

        self._log_creation()

    def sample_action(self):
        ac = self.ac_space.sample()
        ac['default'] = np.array([ac['default']])
        return ac

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('creating a meta sac agent')
            for i, _actor in enumerate(self._actors):
                logger.info('Actor #{} has %d parameters'.format(i + 1), count_parameters(_actor))
            logger.info('the critic1 has %d parameters', count_parameters(self._critic1))
            logger.info('the critic2 has %d parameters', count_parameters(self._critic2))

    def sync_networks(self):
        if self._config.meta_update_target == 'HL' or \
           self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass

    def act(self, ob, is_train=True):
        if self._config.hrl:
            return self._actors[0].act(ob, is_train, return_log_prob=True)
        else:
            return [0], None, None

    def act_log(self, ob):
        return self._actors[0].act_log(ob)

    def _update_network(self, transitions, step=0):
        info = {}

        # pre-process observations
        o, o_next = transitions['ob'], transitions['ob_next']
        bs = len(transitions['done'])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        # update alpha
        actions_real, log_pi = self.act_log(o)
        alpha_loss = -(self._log_alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(self._critic1(o, actions_real),
                                self._critic2(o, actions_real)).mean()
        info['entropy_alpha'] = alpha.cpu().item()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
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

        if step % self._config.actor_update_freq == 0:
            # update the actor
            for _actor_optim in self._actor_optims:
                _actor_optim.zero_grad()
            actor_loss.backward()
            for i, _actor in enumerate(self._actors):
                sync_grads(_actor)
                self._actor_optims[i].step()

        # update the critic
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        sync_grads(self._critic2)
        self._critic2_optim.step()

        # include info from policy
        if len(self._actors) == 1:
            info.update(self._actors[0].info)
        else:
            constructed_info = {}
            for i, _agent in enumerate(self._actors):
                for j, _actor in enumerate(_agent):
                    for k, v in _actor.info:
                        constructed_info['agent_{}/skill_{}/{}'.format(i + 1, j + 1, k)] = v
            info.update(constructed_info)

        return mpi_average(info)
