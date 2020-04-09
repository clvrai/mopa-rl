import os
from collections import OrderedDict

import numpy as np
import torch

from rl.PPO_agent import SACAgent
from rl.normalizer import Normalizer
from rl.mp_agent import MpAgent
from util.logger import logger
from util.pytorch import to_tensor, get_ckpt_path
from util.gym import action_size, observation_size
from env.action_spec import ActionSpec

from gym import spaces

from util.logger import logger

class LowLevelAgent(SACAgent):
    ''' Low level agent that includes skill sets for each agent, their
        execution procedure given observation and skill selections from
        meta-policy, and their training (for single-skill-per-agent cases
        only).
    '''

    def __init__(self, config, ob_space, ac_space, actor, critic, non_limited_idx=None):
        self._non_limited_idx = non_limited_idx
        super().__init__(config, ob_space, ac_space, actor, critic)

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a low-level agent')

    def _build_actor(self, actor):
        config = self._config

        # parse body parts and skills
        self._actors = []
        self._ob_norms = []
        self._planners = []

        # load networks
        #mp = MpAgent(config, ac_space, non_limited_idx)

        # Change here !!!!!!
        if config.primitive_skills:
            skills = config.primitive_skills
        else:
            skills = ['primitive']

        self._skills = skills
        planner_i = 0

        for skill in skills:
            skill_actor = actor(config, self._ob_space, self._ac_space, config.tanh_policy)
            skill_ob_norm = Normalizer(self._ob_space,
                                       default_clip_range=config.clip_range,
                                       clip_obs=config.clip_obs)

            if self._config.meta_update_target == 'HL':
                if "mp" not in skill:
                    path = os.path.join(config.primitive_dir, skill)
                    ckpt_path, ckpt_num = get_ckpt_path(path, None)
                    logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
                    ckpt = torch.load(ckpt_path)

                    if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
                        # backward compatibility to older checkpoints
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
                    else:
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0])
                    skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

            if 'mp' in skill:
                ignored_contacts = config.ignored_contact_geom_ids[planner_i]
                planner = MpAgent(config, self._ac_space, self._non_limited_idx, ignored_contacts)
                self._planners.append(planner)
                planner_i += 1
            else:
                self._planners.append(None)

            skill_actor.to(config.device)
            self._actors.append(skill_actor)
            self._ob_norms.append(skill_ob_norm)

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
            return traj, success, target_qpos, ac
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

    def return_skill_type(self, meta_ac):
        skill_idx = int(meta_ac['default'][0])
        return self._skills[skill_idx]

    def act_log(self, ob, meta_ac=None):
        ''' Note: only usable for SAC agents '''
        if len(meta_ac['default']) == 1:
            skill_idx = int(meta_ac['default'][0])
            return self._actors[skill_idx].act_log(ob)
        else:
            actions = torch.zeros(len(meta_ac['default']), action_size(self._ac_space)).to(self._config.device)
            log_pis = torch.zeros_like(meta_ac['default']).to(self._config.device)
            for i in range(len(self._skills)):
                ac, log_pi = self._actors[i].act_log(ob)
                actions +=  ac['default'] * (meta_ac['default'] == i).float()
                log_pis += log_pi * (meta_ac['default'] == i).float()

            return OrderedDict([('default', actions)]), log_pis

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
           self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass

    def curr_pos(self, env, meta_ac):
        skill = self.return_skill_type(meta_ac)

        import pdb
        pdb.set_trace()
        if 'mp' in skill:
            return env._get_pos('fingertip')[:env.sim.model.nu].copy()
        else:
            return env._get_pos('box')[:env.sim.model.nu].copy()

