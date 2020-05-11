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
        self._history = defaultdict(list)
        return batch


class PlannerRolloutRunner(object):
    def __init__(self, config, env, env_eval, meta_pi, pi):
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._ik_env = gym.make(config.env, **config.__dict__)
        self._meta_pi = meta_pi
        self._pi = pi


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
            ob = env.reset()

            # run rollout
            meta_ac = None
            counter = {'mp': 0, 'rl': 0}
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

                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if random_exploration: # Random exploration for SAC
                        ac = env.action_space.sample()
                        ac_before_activation = None
                        stds = None
                    else:
                        if config.hrl:
                            ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                        else:
                            ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)


                    curr_qpos = env.sim.data.qpos.copy()
                    prev_ob = ob.copy()
                    if pi.is_planner_ac(ac):
                        counter['mp'] += 1
                        target_qpos = curr_qpos.copy()
                        target_qpos[env.ref_joint_pos_indexes] += ac['default']
                        traj, success = pi.plan(curr_qpos, target_qpos)
                        if success:
                            for next_qpos in traj:
                                ll_ob = ob.copy()
                                converted_ac = env.form_action(next_qpos)
                                # ac = env.form_action(next_qpos)
                                if config.reuse_data:
                                    inter_subgoal_ac = OrderedDict([('default', next_qpos[env.ref_joint_pos_indexes] - env.sim.data.qpos[env.ref_joint_pos_indexes].copy())])
                                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': inter_subgoal_ac, 'ac_before_activation': ac_before_activation})
                                ob, reward, done, info = env.step(converted_ac, is_planner=True)
                                # ob, reward, done, info = env.step(ac, is_planner=True)
                                if config.reuse_data:
                                    rollout.add({'done': done, 'rew': reward})
                                meta_rew += reward
                                ep_len += 1
                                step += 1
                                ep_rew += reward
                                meta_len += 1
                                reward_info.add(info)
                                if every_steps is not None and step % every_steps == 0 and config.reuse_data:
                                    # last frame
                                    ll_ob = ob.copy()
                                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                                    yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
                                if done or ep_len >= max_step:
                                    break
                            if self._config.subgoal_hindsight: # refer to HAC
                                hindsight_subgoal_ac = OrderedDict([('default', env.sim.data.qpos[env.ref_joint_pos_indexes].copy() - curr_qpos[env.ref_joint_pos_indexes])])
                                rollout.add({'ob': prev_ob, 'meta_ac': meta_ac, 'ac': hindsight_subgoal_ac, 'ac_before_activation': ac_before_activation})
                            else:
                                rollout.add({'ob': prev_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                            rollout.add({'done': done, 'rew': meta_rew})
                            if every_steps is not None and step % every_steps == 0:
                                # last frame
                                ll_ob = ob.copy()
                                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
                        else:
                            ll_ob = ob.copy()
                            meta_rollout.add({
                                'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                            })
                            # reward = self._config.invalid_planner_rew
                            reward, _  = env.compute_reward(np.zeros(env.sim.model.nu))
                            reward += self._config.invalid_planner_rew
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                            done, info, _ = env._after_step(reward, False, {})
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
                                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
                    else:
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                        counter['rl'] += 1
                        ob, reward, done, info = env.step(ac)
                        rollout.add({'done': done, 'rew': reward})
                        ep_len += 1
                        step += 1
                        ep_rew += reward
                        meta_len += 1
                        meta_rew += reward
                        reward_info.add(info)
                        if every_steps is not None and step % every_steps == 0:
                            # last frame
                            ll_ob = ob.copy()
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                            yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
                reward_info.add({'meta_rew': meta_rew})
                if every_steps is not None and step % every_steps == 0 and config.meta_update_target == 'HL':
                    ll_ob = ob.copy()
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                    meta_rollout.add({'meta_ob': ob})
                    yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
            ep_info.add({'len': ep_len, 'rew': ep_rew})
            ep_info.add(counter)
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info('Ep %d rollout: %s %s', episode,
                        {k: v for k, v in reward_info_dict.items()
                         if not 'qpos' in k and np.isscalar(v)}, {k: v for k, v in counter.items()})

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                ll_ob = ob.copy()
                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                meta_rollout.add({'meta_ob': ob})
                yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)


    def run_episode(self, max_step=10000, is_train=True, record=False, random_exploration=False):
        config = self._config
        device = config.device
        env = self._env if is_train else self._env_eval
        ik_env = self._ik_env
        ik_env.reset()
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = Info()
        ep_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ob = env.reset()
        self._record_frames = []
        if record: self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        counter = {'mp': 0, 'rl': 0}
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

            while not done and ep_len < max_step:
                ll_ob = ob.copy()
                if random_exploration: # Random exploration for SAC
                    ac = env.action_space.sample()
                    ac_before_activation = None
                    stds = None
                else:
                    if config.hrl:
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                curr_qpos = env.sim.data.qpos.copy()
                prev_joint_qpos = curr_qpos[env.ref_joint_pos_indexes]
                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                if pi.is_planner_ac(ac):
                    counter['mp'] += 1
                    target_qpos = curr_qpos.copy()
                    target_qpos[env.ref_joint_pos_indexes] += ac['default']
                    traj, success = pi.plan(curr_qpos, target_qpos)
                    ik_env.set_state(target_qpos, env.sim.data.qvel.ravel().copy())
                    goal_xpos, goal_xquat = self._get_mp_body_pos(ik_env, postfix='goal')
                    if success:
                        for next_qpos in traj:
                            ll_ob = ob.copy()
                            converted_ac = env.form_action(next_qpos)
                            # ac = env.form_action(next_qpos)
                            ob, reward, done, info = env.step(converted_ac, is_planner=True)
                            # ob, reward, done, info = env.step(ac, is_planner=True)
                            meta_rew += reward
                            ep_len += 1
                            ep_rew += reward
                            meta_len += 1
                            reward_info.add(info)

                            if record:
                                frame_info = info.copy()
                                frame_info['ac'] = ac['default']
                                frame_info['converted_ac'] = converted_ac['default']
                                frame_info['target_qpos'] = target_qpos
                                frame_info['states'] = 'Valid states'
                                curr_qpos = env.sim.data.qpos.copy()
                                frame_info['curr_qpos'] = curr_qpos
                                frame_info['mp_path_qpos'] = next_qpos[env.ref_joint_pos_indexes]
                                frame_info['goal'] = env.goal
                            ik_qpos = env.sim.data.qpos.ravel().copy()
                            ik_qpos[env.ref_joint_pos_indexes] = next_qpos[env.ref_joint_pos_indexes]
                            ik_env.set_state(ik_qpos, ik_env.sim.data.qvel.ravel())
                            xpos, xquat = self._get_mp_body_pos(ik_env)
                            vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                            self._store_frame(env, frame_info, None, vis_pos=vis_pos, planner=True)
                            if done or ep_len >= max_step:
                                break
                        rollout.add({'done': done, 'rew': meta_rew})
                    else:
                        # reward = self._config.invalid_planner_rew
                        reward, _ = env.compute_reward(np.zeros(env.sim.model.nu))
                        reward += self._config.invalid_planner_rew
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': None})
                        done, info, _ = env._after_step(reward, False, {})
                        rollout.add({'done': done, 'rew': reward})
                        ep_len += 1
                        ep_rew += reward
                        meta_len += 1
                        reward_info.add(info)
                        if record:
                            frame_info = info.copy()
                            frame_info['states'] = 'Invalid states'
                            frame_info['target_qpos'] = target_qpos
                            curr_qpos = env.sim.data.qpos.copy()
                            frame_info['curr_qpos'] = curr_qpos
                            frame_info['goal'] = env.goal
                            frame_info['contacts'] = env.sim.data.ncon

                            xpos, xquat = self._get_mp_body_pos(ik_env)
                            vis_pos = [(xpos, xquat), (goal_xpos, goal_xquat)]
                            self._store_frame(env, frame_info, None, vis_pos=vis_pos, planner=True)
                else:
                    counter['rl'] += 1
                    ob, reward, done, info = env.step(ac)
                    ep_len += 1
                    ep_rew += reward
                    meta_len += 1
                    meta_rew += reward
                    reward_info.add(info)
                    rollout.add({'done': done, 'rew': reward})
                    if record:
                        frame_info = info.copy()
                        frame_info['ac'] = ac['default']
                        frame_info['std'] = np.array(stds['default'].detach().cpu())[0]
                        self._store_frame(env, frame_info)
            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})

        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})

        ep_info.add({'len': ep_len, 'rew': ep_rew})
        ep_info.add(counter)
        reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
        ep_info.add(reward_info_dict)
        # last frame
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

    def _store_frame(self, env, info={}, subgoal=None, vis_pos=[], planner=False):
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

        geom_colors = {}
        if planner:
            for k in env.body_geoms:
                geom_idx = env.sim.model.geom_name2id(k)
                color = env.sim.model.geom_rgba[geom_idx]
                geom_colors[geom_idx] = color.copy()
                color[0] = 0.0
                color[1] = 0.6
                color[2] = 0.4
                env.sim.model.geom_rgba[geom_idx] = color

        frame = env.render('rgb_array') * 255.0
        env._set_color('subgoal', [0.2, 0.9, 0.2, 0.])
        for xpos, xquat in vis_pos:
            if xpos is not None and xquat is not None:
                for k in xpos.keys():
                    color = env._get_color(k)
                    color[-1] = 0.
                    env._set_color(k, color)

        if planner:
            for geom_idx, color in geom_colors.items():
                env.sim.model.geom_rgba[geom_idx] = color

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

