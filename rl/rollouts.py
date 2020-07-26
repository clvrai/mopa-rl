import os
from collections import defaultdict

import numpy as np
import torch
import cv2
import gym
from collections import OrderedDict
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from util.logger import logger
from util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
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
        ik_env = self._ik_env if config.use_ik_target else None
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
            if config.use_ik_target:
                ik_env.reset()

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

                while not done and ep_len < max_step and meta_len < config.max_meta_len:
                    ll_ob = ob.copy()
                    if random_exploration: # Random exploration for SAC
                        ac = pi._actors[0]._ac_space.sample()
                        ac_before_activation = None
                        stds = None
                    else:
                        if config.hrl:
                            ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                        else:
                            ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})

                    if config.use_ik_target:
                        curr_qpos = env.sim.data.qpos.copy()
                        target_cart = np.clip(env.sim.data.get_site_xpos(config.ik_target)[:len(env.min_world_size)] + config.action_range * ac['default'], env.min_world_size, env.max_world_size)
                        if len(env.min_world_size) == 2:
                            target_cart = np.concatenate((target_cart, np.array([env.sim.data.get_site_xpos(config.ik_target)[2]])))

                        if 'quat' in ac.keys():
                            target_quat = mat2quat(env.sim.data.get_site_xmat(config.ik_target))
                            target_quat = target_quat[[3, 0, 1, 1]]
                            target_quat = quat_mul(target_quat, (ac['quat']/np.linalg.norm(ac['quat'])).astype(np.float64))
                        else:
                            target_quat = None
                        ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
                        result = qpos_from_site_pose(ik_env, config.ik_target, target_pos=target_cart, target_quat=target_quat, rot_weight=2.,
                                      joint_names=env.robot_joints, max_steps=40, tol=1e-2)
                        target_qpos = env.sim.data.qpos.copy()
                        target_qpos[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes].copy()
                        pre_converted_ac = (target_qpos[env.ref_joint_pos_indexes]-curr_qpos[env.ref_joint_pos_indexes])/env._ac_scale
                        if 'gripper' in ac.keys():
                            pre_converted_ac = np.concatenate((pre_converted_ac, ac['gripper']))
                        converted_ac = OrderedDict([('default', pre_converted_ac)])
                        ob, reward, done, info = env.step(converted_ac)
                    else:
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
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            logger.info('Ep %d rollout: %s', episode,
                        {k: v for k, v in reward_info_dict.items()
                         if not 'qpos' in k and np.isscalar(v)})

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
        ik_env = self._ik_env if config.use_ik_target else None
        meta_pi = self._meta_pi
        pi = self._pi

        rollout = Rollout()
        meta_rollout = MetaRollout()
        reward_info = defaultdict(list)

        done = False
        ep_len = 0
        ep_rew = 0
        ob = env.reset()
        if config.use_ik_target:
            ik_env.reset()
        self._record_frames = []
        if record: self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        # run rollout
        meta_ac = None
        total_contact_force = 0.
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

                rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                if config.use_ik_target:
                    curr_qpos = env.sim.data.qpos.copy()
                    target_cart = np.clip(env.sim.data.get_site_xpos(config.ik_target)[:len(env.min_world_size)] + config.action_range * ac['default'], env.min_world_size, env.max_world_size)
                    if len(env.min_world_size) == 2:
                        target_cart = np.concatenate((target_cart, np.array([env.sim.data.get_site_xpos(config.ik_target)[2]])))

                    if 'quat' in ac.keys():
                        target_quat = mat2quat(env.sim.data.get_site_xmat(config.ik_target))
                        target_quat = target_quat[[3, 0, 1, 1]]
                        target_quat = quat_mul(target_quat, (ac['quat']/np.linalg.norm(ac['quat'])).astype(np.float64))
                    else:
                        target_quat = None
                    ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
                    result = qpos_from_site_pose(ik_env, config.ik_target, target_pos=target_cart, target_quat=target_quat, rot_weight=2.,
                                  joint_names=env.robot_joints, max_steps=40, tol=1e-3)
                    target_qpos = env.sim.data.qpos.copy()
                    target_qpos[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes].copy()
                    pre_converted_ac = (target_qpos[env.ref_joint_pos_indexes]-curr_qpos[env.ref_joint_pos_indexes])/env._ac_scale
                    if 'gripper' in ac.keys():
                        pre_converted_ac = np.concatenate((pre_converted_ac, ac['gripper']))
                    converted_ac = OrderedDict([('default', pre_converted_ac)])

                    ob, reward, done, info = env.step(converted_ac)
                else:
                    ob, reward, done, info = env.step(ac)


                contact_force = env.get_contact_force()
                total_contact_force += contact_force
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
                    frame_info['contact_force'] = contact_force
                    if config.use_ik_target:
                        frame_info['converted_ac'] = converted_ac['default']
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

        ep_info = {'len': ep_len, 'rew': ep_rew, "avg_conntact_force": total_contact_force/ep_len}
        for key, value in reward_info.items():
            if isinstance(value[0], (int, float, bool)):
                if '_mean' in key:
                    ep_info[key] = np.mean(value)
                else:
                    ep_info[key] = np.sum(value)

        return rollout.get(), meta_rollout.get(), ep_info, self._record_frames

    def _store_frame(self, env, info={}):
        color = (200, 200, 200)

        text = "{:4} {}".format(env._episode_length,
                                env._episode_reward)

        frame = env.render('rgb_array') * 255.0

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
