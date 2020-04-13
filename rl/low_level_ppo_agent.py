import os
from collections import OrderedDict

import numpy as np
import torch

from rl.base_agent import BaseAgent
from rl.ppo_agent import PPOAgent
from rl.normalizer import Normalizer
from rl.dataset import LowLevelPPOReplayBuffer, RandomSampler
from rl.mp_agent import MpAgent
from util.logger import logger
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, \
    obs2tensor, to_tensor, get_ckpt_path
from util.gym import action_size, observation_size
from env.action_spec import ActionSpec

from gym import spaces

from util.logger import logger

class LowLevelPPOAgent(BaseAgent):
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

        self._agents = []
        for skill in config.primitive_skills:
            if 'mp' in skill:
                self._agents.append(PPOAgent(config, ob_space, subgoal_space, actor, critic, postfix='_'+skill))
            else:
                self._agents.append(PPOAgent(config, ob_space, ac_space, actor, critic, postfix='_'+skill))

        self._build_planner()
        sampler = RandomSampler()
        self._buffer = LowLevelPPOReplayBuffer(['ob', 'ac', 'meta_ac', 'done', 'rew', 'ret', 'adv', 'ac_before_activation'],
                                    config.buffer_size,
                                    len(config.primitive_skills),
                                    sampler.sample_func)

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
                ac, activation = self._agents[skill_idx]._actor.act(ob, is_train)
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
                    ac, activation, stds = self._agents[skill_idx]._actor.act(ob, False, return_stds=return_stds)
                else:
                    ac, activation = self._agents[skill_idx]._actor.act(ob, False, return_stds=return_stds)
            else:
                if return_stds:
                    ac, activation, stds = self._agents[skill_idx]._actor.act(ob, is_train, return_stds=return_stds)
                else:
                    ac, activation = self._agents[skill_idx]._actor.act(ob, is_train, return_stds=return_stds)

        if return_stds:
            return ac, activation, stds
        else:
            return ac, activation

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
        vpreds = []
        for i in range(len(self._config.primitive_skills)):
            vpreds.append(self._agents[i]._critic(ob).detach().cpu().numpy()[:, 0])
        # vpred = self._critic(ob).detach().cpu().numpy()[:,0]

        done = rollouts['done']
        rew = rollouts['rew']
        adv = np.empty((T, ) , 'float32')
        lastgaelam = 0
        next_skill_idx = int(rollouts['meta_ac'][T-1]['default'])
        vpred_next = vpreds[next_skill_idx][T]
        vpred = [vpred_next]
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            skill_idx = int(rollouts['meta_ac'][t-1]['default'])
            vpred_curr = vpreds[skill_idx][t]
            vpred.append(vpred_curr)
            delta = rew[t] + self._config.discount_factor * vpred_next * nonterminal - vpred_curr
            adv[t] = lastgaelam = delta + self._config.discount_factor * self._config.gae_lambda * nonterminal * lastgaelam
            v_pred_next = vpred_curr

        vpred.reverse()
        assert len(vpred) == T + 1
        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        rollouts['adv'] = ((adv - adv.mean()) / adv.std()).tolist()
        rollouts['ret'] = ret.tolist()


    def train(self):
        train_info = {}
        for i in range(len(self._agents)):
            self._soft_update_target_network(self._agents[i]._old_actor, self._agents[i]._actor, 0.0)

        for _ in range(self._config.num_batches):
            for skill_idx in range(len(self._config.primitive_skills)):
                if self._buffer._current_size[skill_idx] > 0:
                    transitions = self._buffer.sample(self._config.batch_size, skill_idx)
                else:
                    transitions = self._buffer.create_empty_transition()

                info = self._agents[skill_idx]._update_network(transitions)
                train_info.update(info)
                train_info.update({
                    'actor_grad_norm': compute_gradient_norm(self._agents[i]._actor),
                    'actor_weight_norm': compute_weight_norm(self._agents[i]._actor),
                    'critic_grad_norm': compute_gradient_norm(self._agents[i]._critic),
                    'critic_weight_norm': compute_weight_norm(self._agents[i]._critic),
                })

        self._buffer.clear()

        return train_info

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
           self._config.meta_update_target == 'both':
            for _agent in self._agents:
                _agent.sync_networks()
        else:
            pass
