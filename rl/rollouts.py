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
        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    def __init__(self, config, env, meta_pi, pi):
        self._config = config
        self._env = env
        self._meta_pi = meta_pi
        self._pi = pi
        self._ik_env = gym.make(config.env, **config.__dict__)


    def run_episode(self, max_step=10000, is_train=True, record=False, random_exploration=False):
        config = self._config
        device = config.device
        env = self._env
        ik_env = self._ik_env
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        ik_env.reset()
        ob = self._env.reset()
        self._record_frames = []
        if record: self._store_frame()

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

            if self._config.hrl and 'subgoal' in meta_ac.keys():
                joint_space = env.joint_space['default']
                minimum = joint_space.low
                maximum = joint_space.high

                if self._config.subgoal_type == 'joint':
                    subgoal = curr_qpos[:env.model.nu]+meta_ac['subgoal']
                else:
                    subgoal_cart = meta_ac['subgoal']
                    subgoal_cart = np.clip(subgoal_cart, meta_pi.ac_space['subgoal'].low, meta_pi.ac_space['subgoal'].high)
                    ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], self._env._get_pos('subgoal')[2]])
                    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                          joint_names=env.model.joint_names[:env.model.nu], max_steps=100, trials=30, progress_thresh=10000.)
                    subgoal = result.qpos[:env.model.nu].copy()
                subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

                ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
                goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')
                subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
                target_qpos = np.concatenate([subgoal, env.goal])

                env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()
                if random_exploration:
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
                saved_qpos.append(env.sim.get_state().qpos.copy()) 
                ob, reward, done, info = env.step(ac)


                rollout.add({'done': done, 'rew': reward})
                acs.append(ac)
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

                    self._store_frame(frame_info)
            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})

        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})
        saved_qpos.append(env.sim.get_state().qpos.copy())

        ep_info = {'len': ep_len, 'rew': ep_rew}
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)
        ep_info['saved_qpos'] = saved_qpos

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames


    def run_episode_with_mp(self, max_step=10000, is_train=True, record=False):
        config = self._config
        device = config.device
        env = self._env
        max_step = env.max_episode_steps
        meta_pi = self._meta_pi
        pi = self._pi

        ik_env = self._ik_env
        ik_env.reset()


        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        mp_success = 0

        ob = env.reset()
        self._record_frames = []
        if record: self._store_frame()

        # buffer to save qpos
        saved_qpos = []

        # Run rollout
        meta_ac = None
        success = False
        path_length = []

        skill_count = {}
        if self._config.hrl:
            for skill in pi._skills:
                skill_count[skill] = 0

        while not done and ep_len < max_step:
            meta_ac, meta_ac_before_activation, meta_log_prob =\
                    meta_pi.act(ob, is_train=is_train)

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
            })
            meta_len = 0
            meta_rew = 0

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
                    ik_env._set_pos('subgoal', [subgoal_cart[0], subgoal_cart[1], self._env._get_pos('subgoal')[2]])
                    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=ik_env._get_pos('subgoal'), target_quat=ik_env._get_quat('subgoal'),
                                                          joint_names=env.model.joint_names[:env.model.nu], max_steps=100, trials=30, progress_thresh=20.0)
                    subgoal = result.qpos[:env.model.nu].copy()
                subgoal[env._is_jnt_limited] = np.clip(subgoal[env._is_jnt_limited], minimum[env._is_jnt_limited], maximum[env._is_jnt_limited])

            ik_env.set_state(np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:]]), env.sim.data.qvel.ravel().copy())
            goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')

            # Will change fingertip to variable later
            subgoal_site_pos = ik_env.data.get_site_xpos("fingertip")[:-1].copy()
            target_qpos = np.concatenate([subgoal, env.sim.data.qpos[env.model.nu:].copy()])
            env._set_pos('subgoal', [subgoal_site_pos[0], subgoal_site_pos[1], env._get_pos('subgoal')[2]])

            skill_type = pi.return_skill_type(meta_ac)
            skill_count[skill_type] += 1
            if skill_type == 'mp':
                traj = pi.plan(curr_qpos, target_qpos)
                success = len(np.unique(traj)) != 1 and traj.shape[0] != 1 and ik_env.sim.data.ncon == 0
                if success:
                    mp_success += 1
                    for i, state in enumerate(traj[1:]):
                        ll_ob = ob.copy()
                        if self._config.meta_update_target == 'HL' and self._config.goal_replace:
                            if self._config.subgoal_type == 'joint':
                                ll_ob['goal'] = subgoal_site_pos
                            else:
                                ll_ob['goal'] = subgoal_cart

                        curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                        ac = OrderedDict([('default', state[:env.model.nu] - curr_qpos)])
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': None})
                        saved_qpos.append(env.sim.get_state().qpos.copy())

                        ob, reward, done, info = env.step(ac)

                        rollout.add({'done': done, 'rew': reward})
                        acs.append(ac)
                        ep_len += 1
                        ep_rew += reward
                        meta_len += 1
                        meta_rew += reward

                        for key, value in info.items():
                            reward_info[key].append(value)

                        if record:
                            frame_info = info.copy()
                            frame_info['ac'] = ac['default']
                            frame_info['states'] = 'Valid states'
                            frame_info['curr_qpos'] = curr_qpos
                            frame_info['mp_qpos'] = state[:env.model.nu]
                            frame_info['mp_path_qpos'] = traj[i+1][:env.model.nu]
                            frame_info['goal'] = env.goal
                            frame_info['skill_type'] = skill_type
                            frame_info['meta_subgoal_cart'] = subgoal_site_pos
                            frame_info['meta_subgoal_joint'] = subgoal
                            frame_info['path_length'] = len(traj[1:])
                            for i, k in enumerate(meta_ac.keys()):
                                if k == 'subgoal' and k != 'default':
                                    frame_info['meta_subgoal'] = meta_ac[k]
                                elif k != 'default':
                                    frame_info['meta_'+k] = meta_ac[k]

                            ik_env.set_state(np.concatenate((state[:env.model.nu], env.sim.data.qpos[env.model.nu:])), ik_env.sim.data.qvel.ravel())
                            xpos, xquat = self._get_mp_body_pos(ik_env)
                            vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                            self._store_frame(frame_info, subgoal_site_pos, vis_pos=vis_pos)

                        if done or ep_len >= max_step or meta_len >= config.max_meta_len:
                            break
                    meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
                    reward_info['meta_rew'].append(meta_rew)
                else:
                    for i in range(self._config.max_meta_len):
                        reward = self._config.meta_subgoal_rew
                        ep_len += 1
                        ep_rew += reward
                        meta_len += 1
                        meta_rew += reward

                        info = OrderedDict()
                        done, info, _ = env._after_step(reward, False, info)

                        reward_info['episode_success'].append(False)

                        for key, value in info.items():
                            reward_info[key].append(value)

                        if record:
                            frame_info = OrderedDict()
                            frame_info['status'] = 'Invalid states'
                            frame_info['goal'] = env.goal
                            frame_info['skill_type'] = skill_type
                            frame_info['meta_subgoal_cart'] = subgoal_site_pos
                            frame_info['meta_subgoal_joint'] = subgoal
                            for i, k in enumerate(meta_ac.keys()):
                                if k == 'subgoal' and k != 'default':
                                    frame_info['meta_subgoal'] = meta_ac[k]
                                elif k != 'default':
                                    frame_info['meta_'+k] = meta_ac[k]

                            xpos, xquat = self._get_mp_body_pos(env)
                            vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                            self._store_frame(frame_info, subgoal_site_pos, vis_pos=vis_pos)
                        if done or ep_len >= max_step or meta_len >= config.max_meta_len:
                            break
                    meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
                    reward_info['meta_rew'].append(meta_rew)
            else:
                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if config.hrl:
                        if self._config.meta_update_target == 'HL' and self._config.goal_replace:
                            if self._config.subgoal_type == 'joint':
                                ll_ob['goal'] = subgoal_site_pos
                            else:
                                ll_ob['goal'] = subgoal_cart
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                    curr_qpos = env.sim.data.qpos[:env.model.nu].ravel().copy()
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                    saved_qpos.append(env.sim.get_state().qpos.copy())

                    ob, reward, done, info = env.step(ac)


                    rollout.add({'done': done, 'rew': reward})
                    acs.append(ac)
                    ep_len += 1
                    ep_rew += reward
                    meta_len += 1
                    meta_rew += reward

                    for key, value in info.items():
                        reward_info[key].append(value)
                    if record:
                        frame_info = info.copy()
                        frame_info['ac'] = ac['default']
                        frame_info['states'] = 'Valid states'
                        frame_info['curr_qpos'] = curr_qpos
                        frame_info['goal'] = env.goal
                        frame_info['skill_type'] = skill_type
                        frame_info['meta_subgoal_cart'] = subgoal_site_pos
                        frame_info['meta_subgoal_joint'] = subgoal
                        for i, k in enumerate(meta_ac.keys()):
                            if k == 'subgoal' and k != 'default':
                                frame_info['meta_subgoal'] = meta_ac[k]
                            elif k != 'default':
                                frame_info['meta_'+k] = meta_ac[k]
                        self._store_frame(frame_info, subgoal_site_pos)

                    if done or ep_len >= max_step or meta_len >= config.max_meta_len:
                        break
                meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
                reward_info['meta_rew'].append(meta_rew)
        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})
        saved_qpos.append(env.sim.get_state().qpos.copy())

        #ep_info = {'len': ep_len, 'rew': ep_rew, 'path_length': path_length}
        ep_info = {'len': ep_len, 'rew': ep_rew}
        for key, val in skill_count.items():
            ep_info[key] = val
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)
        ep_info['saved_qpos'] = saved_qpos
        ep_info['mp_success'] = mp_success

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames


    def _get_mp_body_pos(self, ik_env, postfix='dummy'):
        xpos = OrderedDict()
        xquat = OrderedDict()
        for i in range(ik_env.model.nu):
            name = 'body'+str(i)
            body_idx = ik_env.model.body_name2id(name)
            xpos[name+'-'+ postfix] = ik_env.sim.data.body_xpos[body_idx].copy()
            xquat[name+'-'+postfix] = ik_env.sim.data.body_xquat[body_idx].copy()

        return xpos, xquat

    def _store_frame(self, info={}, subgoal=None, vis_pos=[]):
        color = (200, 200, 200)

        text = "{:4} {}".format(self._env._episode_length,
                                self._env._episode_reward)

        if self._config.hl_type == 'subgoal' and subgoal is not None:
            self._env._set_pos('subgoal', [subgoal[0], subgoal[1], self._env._get_pos('subgoal')[2]])
            self._env._set_color('subgoal', [0.2, 0.9, 0.2, 1.])

        for xpos, xquat in vis_pos:
            for k in xpos.keys():
                self._env._set_pos(k, xpos[k])
                self._env._set_quat(k, xquat[k])
                color = self._env._get_color(k)
                color[-1] = 0.3
                self._env._set_color(k, color)

        frame = self._env.render('rgb_array') * 255.0
        self._env._set_color('subgoal', [0.2, 0.9, 0.2, 0.])
        for xpos, xquat in vis_pos:
            for k in xpos.keys():
                color = self._env._get_color(k)
                color[-1] = 0.
                self._env._set_color(k, color)

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
