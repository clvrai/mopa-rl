import os
from collections import defaultdict

import numpy as np
import torch
import cv2
import gym
from collections import OrderedDict
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose

from util.logger import logger


class Rollout(object):
    def __init__(self):
        self._history = defaultdict(list)

    def add(self, data):
        for key, value in data.items():
            self._history[key].append(value)

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
    def __init__(self, config, env, meta_pi, pi, mp=None):
        self._config = config
        self._env = env
        self._meta_pi = meta_pi
        self._pi = pi
        self._mp = mp

    def run_episode(self, max_step=10000, is_train=True, record=False):
        config = self._config
        device = config.device
        env = self._env
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        ob = self._env.reset()
        self._record_frames = []
        if record: self._store_frame()

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        while not done and ep_len < max_step:
            curr_meta_ac, meta_ac_before_activation, meta_log_prob = \
                meta_pi.act(ob, is_train=is_train)

            if meta_ac is None:
                meta_ac = curr_meta_ac

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac':  meta_ac,
                'meta_ac_before_activation': meta_ac_before_activation,
                'meta_log_prob': meta_log_prob,
            })

            meta_len = 0
            meta_rew = 0
            subgoal = meta_ac['default'][-2:] if config.hl_type == 'subgoal' else None
            while not done and ep_len < max_step and meta_len < config.max_meta_len:
                ll_ob = ob.copy()
                meta_tmp_ac = OrderedDict([('default', np.array([0]))])
                if config.hrl:
                    if config.hl_type == 'subgoal':
                        # Change later.... change meta_ac structure (subgoal: [], low_level: [0])
                        ll_ob = OrderedDict([('default', np.concatenate((ll_ob['default'], meta_ac['default'])))])
                    ac, ac_before_activation = pi.act(ll_ob, meta_tmp_ac, is_train=is_train)
                else:
                    ac, ac_before_activation = pi.act(ll_ob, is_train=is_train)

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
                    if config.hrl:
                        frame_info['meta_ac'] = []
                        for i, k in enumerate(meta_ac.keys()):
                            frame_info['meta_ac'].append(meta_pi.subdiv_skills[i][int(meta_ac[k])])

                    self._store_frame(frame_info, subgoal)

            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})

        # last frame
        ll_ob = ob.copy()
        if config.hrl and config.hl_type == 'subgoal':
            # Change later.... change meta_ac structure (subgoal: [], low_level: [0])
            ll_ob = OrderedDict([('default', np.concatenate((ll_ob['default'], meta_ac['default'])))])
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


    def mp_run_episode(self, max_step=10000, is_train=True, record=False):
        config = self._config
        device = config.device
        env = self._env
        ik_env = gym.make(config.env, **config.__dict__)
        ik_env.reset()

        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)
        acs = []

        done = False
        ep_len = 0
        ep_rew = 0
        mp_success = 0
        ob = self._env.reset()
        self._record_frames = []
        if record: self._store_frame()

        saved_qpos = []
        meta_ac = None
        while not done and ep_len < max_step:
            curr_meta_ac, meta_ac_before_activation, meta_log_prob =\
                    meta_pi.act(ob, is_train=is_train)

            if meta_ac is None:
                meta_ac = curr_meta_ac

            meta_rollout.add({
                'meta_ob': ob, 'meta_ac': meta_ac,
                'meta_ac_before_activation': meta_ac_before_activation,
                'meta_log_prob': meta_log_prob,
            })
            meta_len = 0
            meta_rew = 0

            curr_qpos = env.sim.data.qpos
            target_qpos = np.array(curr_meta_ac['default'])
            traj, actions = self._mp.plan(curr_qpos, target_qpos)

            ## Change later
            success = len(np.unique(traj)) != 1 and traj.shape[0] != 1
            subgoal = meta_ac['default'][-2:] if config.hl_type == 'subgoal' else None

            if success:
                mp_success += 1
                for state in traj:
                    ll_ob = ob.copy()
                    if config.hrl and config.hl_type == 'subgoal':
                        # Change later.... change meta_ac structure (subgoal: [], low_level: [0])
                        ll_ob = OrderedDict([('default', np.concatenate((ll_ob['default'], meta_ac['default'])))])
                    ac = -(env.sim.data.qpos[:-2] - state[:-2])*env._frame_skip
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
                        self._store_frame(frame_info, subgoal)

                    if done or ep_len >= max_step and meta_len >= config.max_meta_len:
                        break
            else:
                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()

                    #### TEMP
                    meta_tmp_ac = OrderedDict([('default', np.array([0]))])
                    if config.hrl:
                        if config.hl_type == 'subgoal':
                            # Change later.... change meta_ac structure (subgoal: [], low_level: [0])
                            ll_ob = OrderedDict([('default', np.concatenate((ll_ob['default'], meta_ac['default'])))])
                        ac, ac_before_activation = pi.act(ll_ob, meta_tmp_ac, is_train=is_train)
                    else:
                        ac, ac_before_activation = pi.act(ll_ob, is_train=is_train)

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
                        self._store_frame(frame_info, subgoal)

            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
        # last frame
        ll_ob = ob.copy()
        if config.hrl and config.hl_type == 'subgoal':
            # Change later.... change meta_ac structure (subgoal: [], low_level: [0])
            ll_ob = OrderedDict([('default', np.concatenate((ll_ob['default'], meta_ac['default'])))])
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
        ep_info['mp_success'] = mp_success

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames

    def _store_frame(self, info={}, subgoal=None):
        color = (200, 200, 200)

        text = "{:4} {}".format(self._env._episode_length,
                                self._env._episode_reward)

        if self._config.hl_type == 'subgoal' and subgoal is not None:
            self._env._set_pos('subgoal', [subgoal[0], subgoal[1], self._env._get_pos('subgoal')[2]])
            self._env._set_color('subgoal', [0.2, 0.9, 0.2, 1.])

        frame = self._env.render('rgb_array') * 255.0
        self._env._set_color('subgoal', [0.2, 0.9, 0.2, 0.])

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

