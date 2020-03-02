import os
from collections import OrderedDict

import numpy as np
import torch

from rl.sac_agent import SACAgent
from rl.normalizer import Normalizer
from util.logger import logger
from util.pytorch import to_tensor, get_ckpt_path
from util.gym import action_size, observation_size
from env.action_spec import ActionSpec

from gym import spaces


class LowLevelAgent(SACAgent):
    ''' Low level agent that includes skill sets for each agent, their
        execution procedure given observation and skill selections from
        meta-policy, and their training (for single-skill-per-agent cases
        only).
    '''

    def __init__(self, config, ob_space, ac_space, actor, critic):
        super().__init__(config, ob_space, ac_space, actor, critic)

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a low-level agent')

    def _build_actor(self, actor):
        config = self._config

        # parse body parts and skills
        self._actors = []
        self._ob_norms = []

        # load networks

        # Change here !!!!!!
        if config.primitive_skills:
            skills = config.primitive_skills
        else:
            skills = ['primitive']

        for skill in skills:
            skill_actor = actor(config, self._ob_space, self._ac_space, config.tanh_policy)
            skill_ob_norm = Normalizer(self._ob_space,
                                       default_clip_range=config.clip_range,
                                       clip_obs=config.clip_obs)

            if self._config.meta_update_target == 'HL':
                path = os.path.join(config.primitive_dir, skill)
                ckpt_path, ckpt_num = get_ckpt_path(path, None)
                logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
                ckpt = torch.load(ckpt_path)

                if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
                    # backward compatibility to older checkpoints
                    skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
                else:
                    skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0][0])
                skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

            skill_actor.to(config.device)
            self._actors.append(skill_actor)
            self._ob_norms.append(skill_ob_norm)

    def act(self, ob, meta_ac, is_train=True, return_stds=False):
        ac = OrderedDict()
        activation = OrderedDict()
        if self._config.hrl:
            skill_idx = int(meta_ac['default'][0])
            ob_ = ob.copy()
            if self._config.policy == 'mlp':
                ob_ = self._ob_norms[skill_idx].normalize(ob_)
            # if self._config.hl_type == 'subgoal':
            #     ob_['subgoal'] = self._ob_norms[skill_idx].normalize(ob)

            ob_ = to_tensor(ob_, self._config.device)
            if self._config.meta_update_target == 'HL':
                if return_stds:
                    ac_, activation_, stds = self._actors[skill_idx].act(ob_, False, return_stds=return_stds)
                else:
                    ac_, activation_ = self._actors[skill_idx].act(ob_, False, return_stds=return_stds)
            else:
                if return_stds:
                    ac_, activation_, stds = self._actors[skill_idx].act(ob_, is_train, return_stds=return_stds)
                else:
                    ac_, activation_ = self._actors[skill_idx].act(ob_, is_train, return_stds=return_stds)
            ac.update(ac_)
            activation.update(activation_)


        if return_stds:
            return ac, activation, stds
        else:
            return ac, activation

    def act_log(self, ob, meta_ac=None):
        ''' Note: only usable for SAC agents '''
        ob_detached = { k: v.detach().cpu().numpy() for k, v in ob.items() }

        ac = OrderedDict()
        log_probs = []
        skill_idx = meta_ac['default']
        # assert np.sum(skill_idx.detach().cpu().to(int).numpy()) == 0, "multiple skills not supported"
        skill_idx = 0

        ob_ = ob_detached.copy()
        if self._config.policy == 'mlp':
            ob_ = self._ob_norms[skill_idx].normalize(ob_)
        ob_ = to_tensor(ob_, self._config.device)
        ac_, log_probs_ = self._actors[skill_idx].act_log(ob_)
        ac.update(ac_)
        log_probs.append(log_probs_)

        try:
            log_probs = torch.cat(log_probs, -1).sum(-1, keepdim=True)
        except Exception:
            import pdb; pdb.set_trace()

        return ac, log_probs

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
           self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass
