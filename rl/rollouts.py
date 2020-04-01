import os
from collections import defaultdict

import numpy as np
import torch
import cv2
import gym
from collections import OrderedDict
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from util.logger import logger
from util.env import joint_convert
from util.gym import action_size
from util.info import Info


class Rollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def __len__(self):
        return len(self._history['ob'])

    def get(self):
        batch = {}
        batch['ob'] = self._history['ob']
        batch['ac'] = self._history['ac']
        batch['meta_ac'] = self._history['meta_ac']
        batch['ac_before_activation'] = self._history['ac_before_activation']
        batch['done'] = self._history['done']
        batch['rew'] = self._history['rew']
        self._history = defaultdict(list)
        return batch


class MetaRollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

    def __len__(self):
        return len(self._history['ob'])

    def get(self):
        batch = {}
        batch['ob'] = self._history['meta_ob']
        batch['ac'] = self._history['meta_ac']
        batch['ac_before_activation'] = self._history['meta_ac_before_activation']
        batch['log_prob'] = self._history['meta_log_prob']
        batch['done'] = self._history['meta_done']
        batch['rew'] = self._history['meta_rew']
        batch['ag'] = self._history['ag']
        batch['g'] = self._history['g']
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    def __init__(self, config, env, env_eval, meta_pi, pi):
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._meta_pi = meta_pi
        self._pi = pi
        self._ik_env = gym.make(config.env, **config.__dict__)


    def run(self, max_step=10000, is_train=True, random_exploration=False, every_steps=None, every_episodes=None):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.
        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        ik_env = self._ik_env
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = Info()
        ep_info = Info()

        step = 0
        episode = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            ik_env.reset()
            ob = env.reset()

            # run rollout
            meta_ac = None
            while not done and ep_len < max_step:
                meta_ac, meta_ac_before_activation, meta_log_prob =\
                        meta_pi.act(ob, is_train=is_train)

                meta_rollout.add({
                    'meta_ob': ob, 'meta_ac': meta_ac,
                    'meta_ac_before_activation': meta_ac_before_activation,
                    'meta_log_prob': meta_log_prob,
                })
                meta_len = 0
                meta_rew = 0

                curr_qpos = env.sim.data.qpos.ravel().copy()
                subgoal_site_pos = None

                # Subgoal prediction + IK
                if self._config.hrl and 'subgoal' in meta_ac.keys():
                    joint_space = env.joint_space['default']
                    minimum = joint_space.low
                    maximum = joint_space.high

                    if self._config.subgoal_type == 'joint':
                        subgoal = curr_qpos[:env.model.nu]+meta_ac['subgoal']
                    else: # cart
                        subgoal_cart = meta_ac['subgoal']
                        subgoal_cart = np.clip(subgoal_cart, meta_pi.ac_space['subgoal'].low, meta_pi.ac_space['subgoal'].high)
                        ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], env._get_pos('subgoal')[2]])
                        result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                              joint_names=env.model.joint_names[:env.model.nu], max_steps=150, trials=10, progress_thresh=20.)
                        subgoal = result.qpos[:env.model.nu].copy()
                    subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

                    ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
                    goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')
                    subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
                    target_qpos = np.concatenate([subgoal, env.goal])

                    if self._config.hrl and self._config.subgoal_type == 'cart':
                        subgoal_site_pos = np.array([subgoal_cart[0], subgoal_cart[1]])

                    env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if random_exploration: # Random exploration for SAC
                        ac = env.action_space.sample()
                        ac_before_activation = None
                        stds = None
                    else:
                        if config.hrl:
                            if self._config.meta_update_target == 'HL' and self._config.goal_replace:
                                if self._config.subgoal_type == 'joint':
                                    ll_ob['goal'] = subgoal_site_pos
                                else:
                                    ll_ob['goal'] = subgoal_cart
                            ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                        else:
                            ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                    ob, reward, done, info = env.step(ac)
                    if config.subgoal_reward:
                        subgoal_rew, info = env.compute_subgoal_reward('box', info)
                        reward += subgoal_rew

                    rollout.add({'done': done, 'rew': reward})
                    ep_len += 1
                    step += 1
                    ep_rew += reward
                    meta_len += 1
                    meta_rew += reward
                    reward_info.add(info)
                    if every_steps is not None and step % every_steps == 0 and config.hrl_network_to_update == 'LL':
                        # last frame
                        ll_ob = ob.copy()
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                        yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                ag = env._get_pos("fingertip").copy()
                g = env._get_pos('subgoal').copy()
                meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew, 'ag': ag, 'g': g})
                reward_info.add({'meta_rew': meta_rew})
                if every_steps is not None and step % every_steps == 0 and config.hrl_network_to_update == 'HL':
                    ll_ob = ob.copy()
                    if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                        if self._config.subgoal_type == 'joint':
                            ll_ob['goal'] = subgoal_site_pos
                        else:
                            ll_ob['goal'] = subgoal_cart
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                    meta_rollout.add({'meta_ob': ob, 'ag': ag, 'g': g})
                    yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
            ep_info.add({'len': ep_len, 'rew': ep_rew})
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info('Ep %d rollout: %s', episode,
                        {k: v for k, v in reward_info_dict.items()
                         if not 'qpos' in k and np.isscalar(v)})

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                ll_ob = ob.copy()
                if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                    if self._config.subgoal_type == 'joint':
                        ll_ob['goal'] = subgoal_site_pos
                    else:
                        ll_ob['goal'] = subgoal_cart
                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                meta_rollout.add({'meta_ob': ob, 'ag': ag, 'g': g})
                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)


    def run_episode(self, max_step=10000, is_train=True, record=False, random_exploration=False):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        ik_env = self._ik_env
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)

        done = False
        ep_len = 0
        ep_rew = 0
        ik_env.reset()
        ob = env.reset()
        self._record_frames = []
        if record: self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        while not done and ep_len < max_step:
            meta_ac, meta_ac_before_activation, meta_log_prob =\
                    meta_pi.act(ob, is_train=is_train)

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac': meta_ac,
                'meta_ac_before_activation': meta_ac_before_activation,
                'meta_log_prob': meta_log_prob,
            })
            meta_len = 0
            meta_rew = 0

            curr_qpos = env.sim.data.qpos.ravel().copy()
            subgoal_site_pos = None

            # Subgoal prediction + IK
            if self._config.hrl and 'subgoal' in meta_ac.keys():
                joint_space = env.joint_space['default']
                minimum = joint_space.low
                maximum = joint_space.high

                if self._config.subgoal_type == 'joint':
                    subgoal = curr_qpos[:env.model.nu]+meta_ac['subgoal']
                else: # cart
                    subgoal_cart = meta_ac['subgoal']
                    subgoal_cart = np.clip(subgoal_cart, meta_pi.ac_space['subgoal'].low, meta_pi.ac_space['subgoal'].high)
                    ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], env._get_pos('subgoal')[2]])
                    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                          joint_names=env.model.joint_names[:env.model.nu], max_steps=150, trials=10, progress_thresh=20.)
                    subgoal = result.qpos[:env.model.nu].copy()
                subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

                ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
                goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')
                subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
                target_qpos = np.concatenate([subgoal, env.goal])

                if self._config.hrl and self._config.subgoal_type == 'cart':
                    subgoal_site_pos = np.array([subgoal_cart[0], subgoal_cart[1]])

                env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()
                if random_exploration: # Random exploration for SAC
                    ac = env.action_space.sample()
                    ac_before_activation = None
                    stds = None
                else:
                    if config.hrl:
                        if self._config.meta_update_target == 'HL' and self._config.goal_replace:
                            if self._config.subgoal_type == 'joint':
                                ll_ob['goal'] = subgoal_site_pos
                            else:
                                ll_ob['goal'] = subgoal_cart
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                ob, reward, done, info = env.step(ac)
                if config.subgoal_reward:
                    subgoal_rew, info = env.compute_subgoal_reward('box', info)
                    reward += subgoal_rew

                rollout.add({'done': done, 'rew': reward})
                ep_len += 1
                ep_rew += reward
                meta_len += 1
                meta_rew += reward

                for key, value in info.items():
                    reward_info[key].append(value)
                if record:
                    frame_info = info.copy()
                    frame_info['ac'] = ac['default']
                    frame_info['std'] = np.array(stds['default'].detach().cpu())[0]
                    if config.hrl:
                        i = int(meta_ac['default'])
                        frame_info['meta_ac'] = meta_pi.skills[i]
                        for i, k in enumerate(meta_ac.keys()):
                            if k != 'default':
                                frame_info['meta_'+k] = meta_ac[k]

                    self._store_frame(env, frame_info)
            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})

        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})

        ep_info = {'len': ep_len, 'rew': ep_rew}
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames

    def run_with_mp(self, max_step=10000, is_train=True, random_exploration=False, every_steps=None, every_episodes=None):
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        max_step = env.max_episode_steps
        meta_pi = self._meta_pi
        pi = self._pi
        ik_env = self._ik_env
        ik_env.reset()

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0
        step = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            mp_success = 0
            meta_ac = None
            success = False
            skill_count = {}
            if self._config.hrl:
                for skill in pi._skills:
                    skill_count[skill] = 0

            ob = env.reset()

            while not done and ep_len < max_step:
                if random_exploration: # Random exploration for SAC
                    meta_ac = meta_pi.sample_action()
                    meta_ac_before_activation = None
                    meta_log_prob = None
                else:
                    meta_ac, meta_ac_before_activation, meta_log_prob =\
                            meta_pi.act(ob, is_train=is_train)

                meta_rollout.add({
                    'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                })
                meta_len = 0
                meta_rew = 0
                mp_len = 0

                curr_qpos = env.sim.data.qpos.ravel().copy()

                if self._config.hrl and 'subgoal' in meta_ac.keys():
                    joint_space = env.joint_space['default']
                    minimum = joint_space.low
                    maximum = joint_space.high
                    if self._config.subgoal_type == 'joint':
                        subgoal = curr_qpos[:env.model.nu]+meta_ac['subgoal']
                    else:
                        if config.relative_subgoal:
                            subgoal_cart = pi.curr_pos(env, meta_ac) + meta_ac['subgoal']
                        else:
                            subgoal_cart = meta_ac['subgoal']
                        subgoal_cart = np.clip(subgoal_cart, meta_pi.ac_space['subgoal'].low, meta_pi.ac_space['subgoal'].high)
                        ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], env._get_pos('subgoal')[2]])
                        result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                              joint_names=env.model.joint_names[:env.model.nu], max_steps=150, trials=10, progress_thresh=20.0)
                        subgoal = result.qpos[:env.model.nu].copy()
                    subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

                    ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
                    goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')

                    # Will change fingertip to variable later
                    subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
                    target_qpos = np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:].copy()])
                    if self._config.subgoal_type == 'cart':
                        subgoal_site_pos = np.array([subgoal_cart[0], subgoal_cart[1]])

                    env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

                skill_type = pi.return_skill_type(meta_ac)
                skill_count[skill_type] += 1
                if 'mp' in skill_type: # Use motion planner
                    traj, success = pi.plan(curr_qpos, target_qpos)
                    if success:
                        mp_success += 1

                info = OrderedDict()
                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                        if self._config.subgoal_type == 'joint':
                            ll_ob['goal'] = subgoal_site_pos
                        else:
                            ll_ob['goal'] = subgoal_cart
                    if 'mp' in skill_type:
                        if success:
                            curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                            ac = OrderedDict([('default', traj[meta_len][:env.model.nu] - curr_qpos)])
                            ac_before_activation = None
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                            ob, reward, done, info = env.step(ac)
                            if config.subgoal_reward:
                                subgoal_rew, info = env.compute_subgoal_reward('box', info)
                                reward += subgoal_rew
                            rollout.add({'done': done, 'rew': reward})
                        else:
                            reward = self._config.meta_subgoal_rew
                            done, info, _ = env._after_step(reward, False, info)
                    else:
                        if config.hrl:
                            ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                        else:
                            ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)
                        curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})

                        ob, reward, done, info = env.step(ac)
                        if config.subgoal_reward:
                            subgoal_rew, info = env.compute_subgoal_reward('box', info)
                            reward += subgoal_rew
                        rollout.add({'done': done, 'rew': reward})

                    ep_rew += reward
                    meta_rew += reward

                    reward_info.add(info)

                    meta_len += 1
                    ep_len += 1
                    step += 1

                    if done or ep_len >= max_step or meta_len >= config.max_meta_len:
                        break
                ag = env._get_pos("fingertip").copy()
                g = env._get_pos('subgoal').copy()
                meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew, 'ag': ag, 'g': g})
                reward_info.add({'meta_rew': meta_rew})

                if every_steps is not None and step % every_steps == 0:
                    ll_ob = ob.copy()
                    if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                        if self._config.subgoal_type == 'joint':
                            ll_ob['goal'] = subgoal_site_pos
                        else:
                            ll_ob['goal'] = subgoal_cart
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                    meta_rollout.add({'meta_ob': ob, 'ag': ag, 'g': g})
                    yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

            ep_info.add({'len': ep_len, 'rew': ep_rew, 'mp_success': mp_success})
            ep_info.add(skill_count)
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info('Ep %d rollout: %s %s', episode,
                        {k: v for k, v in reward_info_dict.items()
                         if not 'qpos' in k and np.isscalar(v)}, {k: v for k, v in skill_count.items()})

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                ll_ob = ob.copy()
                if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                    if self._config.subgoal_type == 'joint':
                        ll_ob['goal'] = subgoal_site_pos
                    else:
                        ll_ob['goal'] = subgoal_cart
                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                meta_rollout.add({'meta_ob': ob, 'ag': ag, 'g': g})
                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)



    def run_episode_with_mp(self, max_step=10000, is_train=True, record=False, random_exploration=False):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        max_step = env.max_episode_steps
        meta_pi = self._meta_pi
        pi = self._pi

        ik_env = self._ik_env
        ik_env.reset()


        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        done = False
        ep_len = 0
        ep_rew = 0
        mp_success = 0

        ob = env.reset()
        self._record_frames = []
        if record: self._store_frame(env)

        # Run rollout
        meta_ac = None
        success = False
        path_length = []

        skill_count = {}
        if self._config.hrl:
            for skill in pi._skills:
                skill_count[skill] = 0

        while not done and ep_len < max_step:
            if random_exploration: # Random exploration for SAC
                meta_ac = meta_pi.sample_action()
                meta_ac_before_activation = None
                meta_log_prob = None
            else:
                meta_ac, meta_ac_before_activation, meta_log_prob =\
                        meta_pi.act(ob, is_train=is_train)

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
            })
            meta_len = 0
            meta_rew = 0
            mp_len = 0

            curr_qpos = env.sim.data.qpos.ravel().copy()

            if self._config.hrl and 'subgoal' in meta_ac.keys():
                joint_space = env.joint_space['default']
                minimum = joint_space.low
                maximum = joint_space.high
                if self._config.subgoal_type == 'joint':
                    subgoal = curr_qpos[:env.model.nu]+meta_ac['subgoal']
                else:
                    if config.relative_subgoal:
                        subgoal_cart = pi.curr_pos(env, meta_ac) + meta_ac['subgoal']
                    else:
                        subgoal_cart = meta_ac['subgoal']
                    subgoal_cart = np.clip(subgoal_cart, meta_pi.ac_space['subgoal'].low, meta_pi.ac_space['subgoal'].high)
                    ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], env._get_pos('subgoal')[2]])
                    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                          joint_names=env.model.joint_names[:env.model.nu], max_steps=150, trials=10, progress_thresh=20.0)
                    subgoal = result.qpos[:env.model.nu].copy()
                subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

                ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
                goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')

                # Will change fingertip to variable later
                subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
                target_qpos = np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:].copy()])
                if self._config.subgoal_type == 'cart':
                    subgoal_site_pos = np.array([subgoal_cart[0], subgoal_cart[1]])

                env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

            skill_type = pi.return_skill_type(meta_ac)
            skill_count[skill_type] += 1
            if 'mp' in skill_type: # Use motion planner
                traj, success = pi.plan(curr_qpos, target_qpos)
                if success:
                    mp_success += 1

            info = OrderedDict()
            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()
                if self._config.hrl and self._config.meta_update_target == 'HL' and self._config.goal_replace:
                    if self._config.subgoal_type == 'joint':
                        ll_ob['goal'] = subgoal_site_pos
                    else:
                        ll_ob['goal'] = subgoal_cart
                if 'mp' in skill_type:
                    if success:
                        curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                        ac = OrderedDict([('default', traj[meta_len][:env.model.nu] - curr_qpos)])
                        ac_before_activation = None
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                        ob, reward, done, info = env.step(ac)
                        if config.subgoal_reward:
                            subgoal_rew, info = env.compute_subgoal_reward('box', info)
                            reward += subgoal_rew
                        rollout.add({'done': done, 'rew': reward})
                    else:
                        reward = self._config.meta_subgoal_rew
                        done, info, _ = env._after_step(reward, False, info)
                else:
                    if config.hrl:
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)
                    curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})

                    ob, reward, done, info = env.step(ac)
                    if config.subgoal_reward:
                        subgoal_rew, info = env.compute_subgoal_reward('box', info)
                        reward += subgoal_rew
                    rollout.add({'done': done, 'rew': reward})

                ep_rew += reward
                meta_rew += reward

                for key, value in info.items():
                    reward_info[key].append(value)

                if record:
                    frame_info = info.copy()
                    frame_info['ac'] = ac['default']
                    frame_info['states'] = 'Valid states'
                    frame_info['curr_qpos'] = curr_qpos
                    if 'mp' in skill_type and success:
                        frame_info['mp_path_qpos'] = traj[meta_len][:env.model.nu]
                    frame_info['goal'] = env.goal
                    frame_info['skill_type'] = skill_type
                    frame_info['meta_subgoal_cart'] = subgoal_site_pos
                    frame_info['meta_subgoal_joint'] = subgoal
                    for i, k in enumerate(meta_ac.keys()):
                        if k == 'subgoal' and k != 'default':
                            frame_info['meta_subgoal'] = meta_ac[k]
                        elif k != 'default':
                            frame_info['meta_'+k] = meta_ac[k]

                    vis_pos=[]
                    if 'mp' in skill_type and success:
                        ik_env.set_state(np.concatenate((traj[meta_len][:env.model.nu], env.sim.data.qpos[env.model.nu:])), ik_env.sim.data.qvel.ravel())
                        xpos, xquat = self._get_mp_body_pos(ik_env)
                        vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                    self._store_frame(env, frame_info, subgoal_site_pos, vis_pos=vis_pos)
                meta_len += 1
                ep_len += 1

                if done or ep_len >= max_step or meta_len >= config.max_meta_len:
                    break
            ag = env._get_pos("fingertip").copy()
            g = env._get_pos('subgoal').copy()
            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew, 'ag': ag, 'g': g})
            reward_info['meta_rew'].append(meta_rew)
        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob, 'ag': ag, 'g': g})
        #saved_qpos.append(env.sim.get_state().qpos.copy())

        ep_info = {'len': ep_len, 'rew': ep_rew}
        for key, val in skill_count.items():
            ep_info[key] = val
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)
        #ep_info['saved_qpos'] = saved_qpos
        ep_info['mp_success'] = mp_success

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames

    def run_with_subgoal_predictor(self, max_step=10000, is_train=True, random_exploration=False, every_steps=None, every_episodes=None):
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        max_step = env.max_episode_steps
        meta_pi = self._meta_pi
        pi = self._pi
        ik_env = self._ik_env
        ik_env.reset()

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0
        step = 0

        while True:
            done = False
            ep_len = 0
            ep_rew = 0
            mp_success = 0
            prev_primitive = 0
            cur_primitive = 0
            meta_ac = None
            success = False
            skill_count = {}
            if self._config.hrl:
                for skill in pi._skills:
                    skill_count[skill] = 0

            ob = env.reset()

            while not done and ep_len < max_step:
                if not config.meta_oracle:
                    if random_exploration: # Random exploration for SAC
                        meta_ac = meta_pi.sample_action()
                        meta_ac_before_activation = None
                        meta_log_prob = None
                    else:
                        meta_ac, meta_ac_before_activation, meta_log_prob =\
                                meta_pi.act(ob, is_train=is_train)
                else:
                    prev_primitive = cur_primitive
                    cur_primitive_str = env.get_next_primitive(np.array([prev_primitive]))
                    if cur_primitive_str is not None:
                        cur_primitive = [cur_primitive_str in v.lower() for v in config.primitive_skills].index(True)
                        meta_ac = OrderedDict([('default', np.array([[cur_primitive]]))])
                    else:
                        cur_primitive = prev_primitive
                    meta_ac_before_activation = None
                    meta_log_prob = None


                meta_len = 0
                meta_rew = 0
                mp_len = 0

                curr_qpos = env.sim.data.qpos.ravel().copy()


                skill_type = pi.return_skill_type(meta_ac)
                skill_count[skill_type] += 1
                if 'mp' in skill_type: # Use motion planner
                    traj, success, target_qpos, subgoal_ac = pi.plan(curr_qpos, meta_ac=meta_ac,
                                                                     ob=ob.copy(),
                                                                     random_exploration=random_exploration,
                                                                     ref_joint_pos_indexes=env.ref_joint_pos_indexes)
                    if success:
                        mp_success += 1

                info = OrderedDict()
                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if 'mp' in skill_type:
                        if success:
                            for next_qpos in traj:
                                meta_rollout.add({
                                    'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                                })
                                ll_ob = ob.copy()
                                ac = env.form_action(next_qpos)
                                ac_before_activation = None
                                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': subgoal_ac, 'ac_before_activation': ac_before_activation})
                                ob, reward, done, info = env.step(ac)
                                rollout.add({'done': done, 'rew': reward})
                                ep_len += 1
                                step += 1
                                ep_rew += reward
                                meta_len += 1
                                reward_info.add(info)
                                meta_rollout.add({'meta_done': done, 'meta_rew': reward})

                                if every_steps is not None and step % every_steps == 0:
                                    # last frame
                                    ll_ob = ob.copy()
                                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                                    meta_rollout.add({'meta_ob': ob})
                                    yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                                if done or ep_len >= max_step:
                                    break
                        else:
                            meta_rollout.add({
                                'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                            })
                            reward = self._config.meta_subgoal_rew
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': subgoal_ac, 'ac_before_activation': None})
                            done, info, _ = env._after_step(reward, False, info)
                            rollout.add({'done': done, 'rew': reward})
                            ep_len += 1
                            step += 1
                            ep_rew += reward
                            meta_len += 1
                            reward_info.add(info)
                            meta_rollout.add({'meta_done': done, 'meta_rew': reward})
                            if every_steps is not None and step % every_steps == 0:
                                # last frame
                                ll_ob = ob.copy()
                                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                                meta_rollout.add({'meta_ob': ob})
                                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
                    else:
                        meta_rollout.add({
                            'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                        })
                        if random_exploration: # Random exploration for SAC
                            ac = env.action_space.sample()
                            ac_before_activation = None
                            stds = None
                        else:
                            if config.hrl:
                                ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                            else:
                                ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})

                        ob, reward, done, info = env.step(ac)
                        rollout.add({'done': done, 'rew': reward})

                        ep_len += 1
                        step += 1
                        ep_rew += reward
                        meta_len += 1
                        meta_rew += reward
                        reward_info.add(info)
                        meta_rollout.add({'meta_done': done, 'meta_rew': reward})
                        if every_steps is not None and step % every_steps == 0:
                            # last frame
                            ll_ob = ob.copy()
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                            meta_rollout.add({'meta_ob': ob})
                            yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)


            ep_info.add({'len': ep_len, 'rew': ep_rew, 'mp_success': mp_success})
            ep_info.add(skill_count)
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info('Ep %d rollout: %s %s', episode,
                        {k: v for k, v in reward_info_dict.items()
                         if not 'qpos' in k and np.isscalar(v)}, {k: v for k, v in skill_count.items()})
            episode += 1

    def run_episode_with_subgoal_predictor(self, max_step=10000, is_train=True, record=False, random_exploration=False):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        max_step = env.max_episode_steps
        meta_pi = self._meta_pi
        pi = self._pi
        ik_env = self._ik_env
        ik_env.reset()

        self._record_frames = []

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0
        step = 0

        done = False
        ep_len = 0
        ep_rew = 0
        mp_success = 0
        meta_ac = None
        success = False
        prev_primitive = 0
        cur_primitive = 0
        skill_count = {}
        if self._config.hrl:
            for skill in pi._skills:
                skill_count[skill] = 0

        ob = env.reset()
        if record: self._store_frame(env)

        while not done and ep_len < max_step:
            if not config.meta_oracle:
                if random_exploration: # Random exploration for SAC
                    meta_ac = meta_pi.sample_action()
                    meta_ac_before_activation = None
                    meta_log_prob = None
                else:
                    meta_ac, meta_ac_before_activation, meta_log_prob =\
                            meta_pi.act(ob, is_train=is_train)
            else:
                prev_primitive = cur_primitive
                cur_primitive_str = env.get_next_primitive(np.array([prev_primitive]))
                if cur_primitive_str is not None:
                    cur_primitive = [cur_primitive_str in v.lower() for v in config.primitive_skills].index(True)
                    meta_ac = OrderedDict([('default', np.array([[cur_primitive]]))])
                else:
                    cur_primitive = prev_primitive
                meta_ac_before_activation = None
                meta_log_prob = None

            meta_len = 0
            meta_rew = 0
            mp_len = 0

            curr_qpos = env.sim.data.qpos.ravel().copy()


            skill_type = pi.return_skill_type(meta_ac)
            skill_count[skill_type] += 1
            goal_xpos = None
            goal_xquat = None
            if 'mp' in skill_type: # Use motion planner
                traj, success, target_qpos, subgoal_ac = pi.plan(curr_qpos, meta_ac=meta_ac, ob=ob.copy(),
                                                                 ref_joint_pos_indexes=env.ref_joint_pos_indexes)
                ik_env.set_state(target_qpos, env.sim.data.qvel.ravel().copy())
                goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')
                if success:
                    mp_success += 1

            info = OrderedDict()

            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()
                if 'mp' in skill_type:
                    if success:
                        for next_qpos in traj:
                            meta_rollout.add({
                                'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                            })
                            ll_ob = ob.copy()
                            ac = env.form_action(next_qpos)
                            ac_before_activation = None
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': subgoal_ac, 'ac_before_activation': ac_before_activation})
                            ob, reward, done, info = env.step(ac)
                            rollout.add({'done': done, 'rew': reward})

                            ep_len += 1
                            step += 1
                            ep_rew += reward
                            meta_len += 1
                            reward_info.add(info)
                            meta_rollout.add({'meta_done': done, 'meta_rew': reward})

                            if record:
                                frame_info = info.copy()
                                frame_info['ac'] = ac['default']
                                frame_info['target_qpos'] = target_qpos
                                frame_info['subgoal'] = subgoal_ac
                                frame_info['states'] = 'Valid states'
                                curr_qpos = env.sim.data.qpos.copy()
                                frame_info['curr_qpos'] = curr_qpos
                                frame_info['mp_path_qpos'] = next_qpos[env.ref_joint_pos_indexes]
                                frame_info['goal'] = env.goal
                                frame_info['skill_type'] = skill_type
                                for i, k in enumerate(meta_ac.keys()):
                                    if k == 'subgoal' and k != 'default':
                                        frame_info['meta_subgoal'] = meta_ac[k]
                                    elif k != 'default':
                                        frame_info['meta_'+k] = meta_ac[k]

                                vis_pos=[]
                                if 'mp' in skill_type and success:
                                    ik_qpos = env.sim.data.qpos.ravel().copy()
                                    ik_qpos[env.ref_joint_pos_indexes] = next_qpos[env.ref_joint_pos_indexes]
                                    ik_env.set_state(ik_qpos, ik_env.sim.data.qvel.ravel())
                                    xpos, xquat = self._get_mp_body_pos(ik_env)
                                    vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                                self._store_frame(env, frame_info, None, vis_pos=vis_pos)

                    else:
                        meta_rollout.add({
                            'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                        })
                        reward = self._config.meta_subgoal_rew
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': subgoal_ac, 'ac_before_activation': None})
                        done, info, _ = env._after_step(reward, False, info)
                        rollout.add({'done': done, 'rew': reward})
                        ep_len += 1
                        step += 1
                        ep_rew += reward
                        meta_len += 1
                        reward_info.add(info)
                        meta_rollout.add({'meta_done': done, 'meta_rew': reward})
                        if record:
                            frame_info = info.copy()
                            frame_info['states'] = 'Invalid states'
                            frame_info['target_qpos'] = target_qpos
                            curr_qpos = env.sim.data.qpos.copy()
                            frame_info['curr_qpos'] = curr_qpos
                            frame_info['goal'] = env.goal
                            frame_info['skill_type'] = skill_type
                            for i, k in enumerate(meta_ac.keys()):
                                if k == 'subgoal' and k != 'default':
                                    frame_info['meta_subgoal'] = meta_ac[k]
                                elif k != 'default':
                                    frame_info['meta_'+k] = meta_ac[k]

                            vis_pos=[]
                            self._store_frame(env, frame_info, None, vis_pos=[])
                else:
                    meta_rollout.add({
                        'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                    })
                    if config.hrl:
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})

                    ob, reward, done, info = env.step(ac)
                    rollout.add({'done': done, 'rew': reward})

                    ep_len += 1
                    step += 1
                    ep_rew += reward
                    meta_len += 1
                    meta_rew += reward
                    reward_info.add(info)
                    meta_rollout.add({'meta_done': done, 'meta_rew': reward})
                    if record:
                        frame_info = info.copy()
                        frame_info['ac'] = ac['default']
                        curr_qpos = env.sim.data.qpos.copy()
                        frame_info['curr_qpos'] = curr_qpos
                        frame_info['goal'] = env.goal
                        frame_info['skill_type'] = skill_type
                        for i, k in enumerate(meta_ac.keys()):
                            if k == 'subgoal' and k != 'default':
                                frame_info['meta_subgoal'] = meta_ac[k]
                            elif k != 'default':
                                frame_info['meta_'+k] = meta_ac[k]

                        vis_pos=[]
                        self._store_frame(env, frame_info, None, vis_pos=[])


        ep_info.add({'len': ep_len, 'rew': ep_rew, 'mp_success': mp_success})
        ep_info.add(skill_count)
        reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
        ep_info.add(reward_info_dict)
        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})
        return rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True), self._record_frames

    def _get_mp_body_pos(self, ik_env, postfix='dummy'):
        xpos = OrderedDict()
        xquat = OrderedDict()
        for i in range(len(ik_env.ref_joint_pos_indexes)):
            name = 'body'+str(i)
            body_idx = ik_env.sim.model.body_name2id(name)
            xpos[name+'-'+ postfix] = ik_env.sim.data.body_xpos[body_idx].copy()
            xquat[name+'-'+postfix] = ik_env.sim.data.body_xquat[body_idx].copy()

        return xpos, xquat

    def _store_frame(self, env, info={}, subgoal=None, vis_pos=[]):
        color = (200, 200, 200)

        text = "{:4} {}".format(env._episode_length,
                                env._episode_reward)

        if self._config.hl_type == 'subgoal' and subgoal is not None:
            env._set_pos('subgoal', [subgoal[0], subgoal[1], env._get_pos('subgoal')[2]])
            env._set_color('subgoal', [0.2, 0.9, 0.2, 1.])

        for xpos, xquat in vis_pos:
            for k in xpos.keys():
                env._set_pos(k, xpos[k])
                env._set_quat(k, xquat[k])
                color = env._get_color(k)
                color[-1] = 0.3
                env._set_color(k, color)

        frame = env.render('rgb_array') * 255.0
        env._set_color('subgoal', [0.2, 0.9, 0.2, 0.])
        for xpos, xquat in vis_pos:
            if xpos is not None and xquat is not None:
                for k in xpos.keys():
                    color = env._get_color(k)
                    color[-1] = 0.
                    env._set_color(k, color)

        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

        if self._config.record_caption:
            font_size = 0.4
            thickness = 1
            offset = 12
            x, y = 5, fheight + 10
            cv2.putText(frame, text,
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 0), thickness, cv2.LINE_AA)
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = '{}: '.format(k)
                (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, thickness)

                cv2.putText(frame, key_text,
                            (x, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (66, 133, 244), thickness, cv2.LINE_AA)

                cv2.putText(frame, str(v),
                            (x + key_width, y + offset * (i + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, (255, 255, 255), thickness, cv2.LINE_AA)

        self._record_frames.append(frame)
