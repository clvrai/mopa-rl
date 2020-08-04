import os
from collections import defaultdict

import numpy as np
import torch
import cv2
import gym
from gym import spaces
from collections import OrderedDict
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from util.logger import logger
from util.env import joint_convert, mat2quat, quat_mul, quat_inv
from util.gym import action_size
from util.info import Info
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


class Rollout(object):
    def __init__(self): self._history = defaultdict(list)

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
        batch['intra_steps'] = self._history['intra_steps']
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
        # batch['intra_steps'] = self._history['intra_steps']
        self._history = defaultdict(list)
        return batch


class MoPARolloutRunner(object):
    def __init__(self, config, env, env_eval, meta_pi, pi):
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._meta_pi = meta_pi
        self._ik_env = gym.make(config.env, **config.__dict__)
        self._pi = pi


    # @profile
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
            mp_path_len = 0
            interpolation_path_len = 0
            ep_rew_with_penalty = 0
            ob = env.reset()
            if config.use_ik_target:
                ik_env.reset()

            # run rollout
            meta_ac = None
            counter = {'mp': 0, 'rl': 0, 'interpolation': 0, 'mp_fail': 0, 'approximate': 0, 'invalid': 0}
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
                env_step = 0

                ll_ob = ob.copy()
                if random_exploration: # Random exploration for SAC
                    ac = pi._ac_space.sample()
                    for k, space in pi._ac_space.spaces.items():
                        if isinstance(space, spaces.Discrete):
                            ac[k] = np.array([ac[k]])
                    ac_before_activation = None
                    stds = None
                else:
                    if config.hrl:
                        ac, ac_before_activation, stds = pi.act(ll_ob, meta_ac, is_train=is_train, return_stds=True)
                    else:
                        ac, ac_before_activation, stds = pi.act(ll_ob, is_train=is_train, return_stds=True)

                curr_qpos = env.sim.data.qpos.copy()
                prev_qpos = env.sim.data.qpos.copy()
                target_qpos = curr_qpos.copy()
                prev_ob = ob.copy()
                is_planner = False
                if config.discrete_action:
                    is_planner = bool(ac['ac_type'][0])

                if config.use_ik_target:
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
                    result = qpos_from_site_pose(ik_env, config.ik_target, target_pos=target_cart, target_quat=target_quat,
                                  joint_names=env.robot_joints, max_steps=100, tol=1e-2)
                    target_qpos[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes].copy()
                    target_qpos = np.clip(target_qpos, env._jnt_minimum[env.jnt_indices], env._jnt_maximum[env.jnt_indices])
                    displacement = OrderedDict([('default', target_qpos[env.ref_joint_pos_indexes]-curr_qpos[env.ref_joint_pos_indexes])])
                    # inter_subgoal_ac['default'][:len(env.ref_joint_pos_indexes)] = pi.invert_displacement(inter_subgoal_ac['default'][:len(env.ref_joint_pos_indexes)], env._ac_scale)

                if (not config.discrete_action and pi.is_planner_ac(ac) and not config.use_ik_target) or is_planner or (config.use_ik_target and pi.is_planner_ac(displacement)):
                    if not config.use_ik_target:
                        displacement = pi.convert2planner_displacement(ac['default'][:len(env.ref_joint_pos_indexes)], env._ac_scale)
                        target_qpos[env.ref_joint_pos_indexes] += displacement
                        tmp_target_qpos = target_qpos.copy()
                        target_qpos = np.clip(target_qpos, env._jnt_minimum[env.jnt_indices], env._jnt_maximum[env.jnt_indices])
                        target_qpos[np.invert(env._is_jnt_limited[env.jnt_indices])] = tmp_target_qpos[np.invert(env._is_jnt_limited[env.jnt_indices])]

                    if config.find_collision_free and not pi.isValidState(target_qpos):
                        trial = 0
                        while not pi.isValidState(target_qpos) and trial < config.num_trials:
                            d = curr_qpos-target_qpos
                            target_qpos += config.step_size * d/np.linalg.norm(d)
                            trial+=1

                    if pi.isValidState(target_qpos):
                        traj, success, interpolation, valid, exact = pi.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)
                    else:
                        success = False
                        valid = False
                        exact = True

                    if success:
                        if interpolation:
                            counter['interpolation'] += 1
                            interpolation_path_len += len(traj)
                        else:
                            counter['mp'] += 1
                            mp_path_len += len(traj)

                        rew_list = []
                        meta_rew_list = []
                        ob_list = []
                        done_list = []
                        cum_discount = 0
                        cum_discount_list = []
                        cart_list = []
                        quat_list = []
                        for i, next_qpos in enumerate(traj):
                            ll_ob = ob.copy()
                            converted_ac = env.form_action(next_qpos)
                            if i == len(traj)-1:
                                if not config.use_ik_target:
                                    converted_ac['default'][len(env.ref_joint_pos_indexes):] = ac['default'][len(env.ref_joint_pos_indexes):]
                                else:
                                    if 'gripper' in ac.keys():
                                        converted_ac['default'][len(env.ref_joint_pos_indexes):] = ac['gripper']
                            ob, reward, done, info = env.step(converted_ac, is_planner=True)

                            if self._config.use_discount_meta:
                                meta_rew += (config.discount_factor**i) * reward # the last reward is more important
                            else:
                                meta_rew += reward
                            cum_discount += config.discount_factor**i
                            cum_discount_list.append(cum_discount)
                            done_list.append(done)
                            meta_rew_list.append(meta_rew)
                            rew_list.append(reward)
                            ob_list.append(ob.copy())
                            if config.use_ik_target:
                                cart_list.append(env.sim.data.get_site_xpos(config.ik_target))
                                quat_list.append(mat2quat(env.sim.data.get_site_xmat(config.ik_target)))
                            ep_len += 1
                            step += 1
                            meta_len += 1
                            env_step += 1
                            ep_rew += reward
                            ep_rew_with_penalty += reward
                            reward_info.add(info)
                            if done or ep_len >= max_step:
                                break
                        rollout.add({'ob': prev_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                        rollout.add({'done': done, 'rew': meta_rew, 'intra_steps': i})

                        if every_steps is not None and step % every_steps == 0:
                            # last frame
                            ll_ob = ob.copy()
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                            ep_info.add({'env_step': env_step})
                            env_step = 0
                            yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                        if config.reuse_data and len(ob_list) > config.min_reuse_span+2:
                            pairs = []
                            for _ in range(min(len(ob_list), config.max_reuse_data)):
                                start = np.random.randint(low=0, high=len(ob_list)-config.min_reuse_span-1)
                                if start + config.min_reuse_span > len(ob_list)-1:
                                    continue
                                goal = np.random.randint(low=start+config.min_reuse_span, high=len(ob_list))
                                if (start, goal) in pairs:
                                    continue

                                pairs.append((start, goal))
                                if not config.use_ik_target:
                                    inter_subgoal_ac = env.form_action(traj[goal], traj[start])
                                    inter_subgoal_ac['default'][:len(env.ref_joint_pos_indexes)] = pi.invert_displacement(inter_subgoal_ac['default'][:len(env.ref_joint_pos_indexes)], env._ac_scale)
                                else:
                                    inter_subgoal_ac = OrderedDict([('default', (cart_list[goal]-cart_list[start])/config.action_range)])
                                    if 'quat' in ac.keys():
                                        inter_subgoal_ac['quat'] = quat_mul(quat_inv(quat_list[start]), quat_list[goal])
                                if pi.is_planner_ac(inter_subgoal_ac) and pi.valid_action(inter_subgoal_ac):
                                    if config.discrete_action:
                                        inter_subgoal_ac['ac_type'] = ac['ac_type']
                                    rollout.add({'ob': ob_list[start], 'meta_ac': meta_ac, 'ac': inter_subgoal_ac, 'ac_before_activation': ac_before_activation})
                                    inter_rew = meta_rew_list[goal] - meta_rew_list[start]
                                    if config.use_discount_meta:
                                        inter_rew *= (config.discount_factor ** (-(start+1)))
                                    rollout.add({'done': done_list[goal], 'rew': inter_rew, 'intra_steps': goal-start-1})

                                    if every_steps is not None and step % every_steps == 0:
                                        # last frame
                                        rollout.add({'ob': ob_list[goal], 'meta_ac': meta_ac})
                                        ep_info.add({'env_step': env_step})
                                        env_step = 0
                                        yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                    else:
                        if not exact:
                            counter['approximate'] += 1
                        if not valid:
                            counter['invalid'] += 1
                        counter['mp_fail'] += 1
                        ll_ob = ob.copy()
                        meta_rollout.add({
                            'meta_ob': ob, 'meta_ac': meta_ac, 'meta_ac_before_activation': meta_ac_before_activation, 'meta_log_prob': meta_log_prob,
                        })
                        reward = 0
                        if config.add_curr_rew:
                            reward, info  = env.compute_reward(np.zeros(env.sim.model.nu))
                            ep_rew += reward
                            reward += self._config.invalid_planner_rew
                        else:
                            reward += self._config.invalid_planner_rew
                            ep_rew += reward
                            info = {}
                        ep_rew_with_penalty += reward
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                        done, info, _ = env._after_step(reward, env._terminal, info)
                        meta_rew += reward
                        rollout.add({'done': done, 'rew': reward, 'intra_steps': 0})
                        ep_len += 1
                        step += 1
                        meta_len += 1
                        env_step += 1
                        reward_info.add(info)
                        meta_rollout.add({'meta_done': done, 'meta_rew': reward})
                        if every_steps is not None and step % every_steps == 0:
                            # last frame
                            ll_ob = ob.copy()
                            rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                            ep_info.add({'env_step': env_step})
                            env_step = 0
                            yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)
                else:
                    ll_ob = ob.copy()
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                    counter['rl'] += 1
                    if not config.use_ik_target:
                        rescaled_ac = OrderedDict([('default', ac['default'].copy())])
                        if not config.discrete_action:
                            rescaled_ac['default'][:len(env.ref_joint_pos_indexes)] /=  config.omega
                        ob, reward, done, info = env.step(rescaled_ac)
                    else:
                        displacement['default'] /= config.omega
                        if 'gripper' in ac.keys():
                            displacement['default'] = np.concatenate((displacement['default'], ac['gripper']))
                        ob, reward, done, info = env.step(displacement)
                    rollout.add({'done': done, 'rew': reward, 'intra_steps': 0})
                    ep_len += 1
                    step += 1
                    ep_rew += reward
                    ep_rew_with_penalty += reward
                    meta_len += 1
                    env_step += 1
                    meta_rew += reward
                    reward_info.add(info)
                    if every_steps is not None and step % every_steps == 0:
                        # last frame
                        ll_ob = ob.copy()
                        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
                        ep_info.add({'env_step': env_step})
                        env_step = 0
                        yield rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True)

                env._reset_prev_state()
                meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
                reward_info.add({'meta_rew': meta_rew})
            ep_info.add({'len': ep_len, 'rew': ep_rew, 'rew_with_penalty': ep_rew_with_penalty})
            if counter['mp'] > 0:
                ep_info.add({"mp_path_len": mp_path_len/counter['mp']})
            if counter['interpolation'] > 0:
                ep_info.add({'interpolation_path_len': interpolation_path_len/counter['interpolation']})
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
                ep_info.add({'env_step': env_step})
                env_step = 0
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
        reward_info = Info()
        ep_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ep_rew_with_penalty = 0
        ob = env.reset()
        if config.use_ik_target:
            ik_env.reset()
        self._record_frames = []
        if record: self._store_frame(env)

        # buffer to save qpos
        saved_qpos = []

        if config.stochastic_eval and not is_train:
            is_train = True

        stochastic = is_train or not config.stochastic_eval
        # run rollout
        meta_ac = None
        total_contact_force = 0.
        counter = {'mp': 0, 'rl': 0, 'interpolation': 0, 'mp_fail': 0, 'approximate': 0, 'invalid': 0}
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
            prev_qpos = env.sim.data.qpos.copy()
            prev_joint_qpos = curr_qpos[env.ref_joint_pos_indexes]
            prev_ob = ob.copy()
            is_planner = False
            target_qpos = env.sim.data.qpos.copy()
            if config.discrete_action:
                is_planner = bool(ac['ac_type'][0])

            if config.use_ik_target:
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
                result = qpos_from_site_pose(ik_env, config.ik_target, target_pos=target_cart, target_quat=target_quat,
                              joint_names=env.robot_joints, max_steps=100, tol=1e-2)
                target_qpos[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes].copy()
                target_qpos = np.clip(target_qpos, env._jnt_minimum[env.jnt_indices], env._jnt_maximum[env.jnt_indices])
                displacement = OrderedDict([('default', target_qpos[env.ref_joint_pos_indexes]-curr_qpos[env.ref_joint_pos_indexes])])

            if (not config.discrete_action and pi.is_planner_ac(ac) and not config.use_ik_target) or is_planner or (config.use_ik_target and pi.is_planner_ac(displacement)):
                if not config.use_ik_target:
                    displacement = pi.convert2planner_displacement(ac['default'][:len(env.ref_joint_pos_indexes)], env._ac_scale)
                    target_qpos[env.ref_joint_pos_indexes] += displacement
                    tmp_target_qpos = target_qpos.copy()
                    target_qpos = np.clip(target_qpos, env._jnt_minimum[env.jnt_indices], env._jnt_maximum[env.jnt_indices])
                    target_qpos[np.invert(env._is_jnt_limited[env.jnt_indices])] = tmp_target_qpos[np.invert(env._is_jnt_limited[env.jnt_indices])]

                if config.find_collision_free and not pi.isValidState(target_qpos):
                    trial = 0
                    while not pi.isValidState(target_qpos) and trial < config.num_trials:
                        d = curr_qpos-target_qpos
                        target_qpos += config.step_size * d/np.linalg.norm(d)
                        trial+=1

                if pi.isValidState(target_qpos):
                    traj, success, interpolation, valid, exact = pi.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)
                else:
                    success = False
                    valid = False
                    exact = True

                env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
                env.color_agent()
                if success:
                    if interpolation:
                        counter['interpolation'] += 1
                    else:
                        counter['mp'] += 1

                    for i, next_qpos in enumerate(traj):
                        ll_ob = ob.copy()
                        converted_ac = env.form_action(next_qpos)
                        if i == len(traj)-1:
                            if not config.use_ik_target:
                                converted_ac['default'][len(env.ref_joint_pos_indexes):] = ac['default'][len(env.ref_joint_pos_indexes):]
                            else:
                                if 'gripper' in ac.keys():
                                    converted_ac['default'][len(env.ref_joint_pos_indexes):] = ac['gripper']
                        ob, reward, done, info = env.step(converted_ac, is_planner=True)
                        contact_force = env.get_contact_force()
                        total_contact_force += contact_force
                        meta_rew += reward
                        ep_len += 1
                        ep_rew += reward
                        ep_rew_with_penalty += reward
                        meta_len += 1
                        reward_info.add(info)

                        if record:
                            frame_info = info.copy()
                            if not config.use_ik_target:
                                frame_info['ac'] = ac['default']
                            else:
                                frame_info['cart'] = ac['default']
                                if 'quat' in ac.keys():
                                    frame_info['quat'] = ac['quat']
                            frame_info['contact_force'] = contact_force
                            frame_info['converted_ac'] = converted_ac['default']
                            frame_info['target_qpos'] = target_qpos
                            frame_info['states'] = 'Valid states'
                            frame_info['std'] = np.array(stds['default'].detach().cpu())[0]
                            curr_qpos = env.sim.data.qpos.copy()
                            frame_info['curr_qpos'] = curr_qpos
                            frame_info['mp_path_qpos'] = next_qpos[env.ref_joint_pos_indexes]
                            if hasattr(env, 'goal'):
                                frame_info['goal'] = env.goal
                            # env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes])
                            self._store_frame(env, frame_info, planner=True)
                        if done or ep_len >= max_step:
                            break
                    rollout.add({'ob': prev_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': ac_before_activation})
                    rollout.add({'done': done, 'rew': meta_rew})
                else:
                    counter['mp_fail'] += 1
                    if not exact:
                        counter['approximate'] += 1
                    elif not valid:
                        counter['invalid'] += 1
                    ll_ob = ob.copy()
                    reward = 0.
                    if config.add_curr_rew:
                        reward, info  = env.compute_reward(np.zeros(env.sim.model.nu))
                        ep_rew += reward
                        reward += self._config.invalid_planner_rew
                    else:
                        reward += self._config.invalid_planner_rew
                        ep_rew += reward
                        info = {}
                    rollout.add({'ob': ll_ob, 'meta_ac': meta_ac, 'ac': ac, 'ac_before_activation': None})
                    done, info, _ = env._after_step(reward, env._terminal, info)
                    rollout.add({'done': done, 'rew': reward})
                    ep_len += 1
                    ep_rew_with_penalty += reward
                    meta_len += 1
                    reward_info.add(info)

                    contact_pairs = []
                    for n in range(env.sim.data.ncon):
                        con = env.sim.data.contact[n]
                        contact_pairs.append((con.geom1, con.geom2))

                    if record:
                        frame_info = info.copy()
                        frame_info['states'] = 'Invalid states'
                        frame_info['target_qpos'] = target_qpos
                        frame_info['contact_pairs'] = contact_pairs
                        if config.use_ik_target:
                            frame_info['cart'] = ac['default']
                            if 'quat' in ac.keys():
                                frame_info['quat'] = ac['quat']
                        frame_info['std'] = np.array(stds['default'].detach().cpu())[0]
                        curr_qpos = env.sim.data.qpos.copy()
                        frame_info['curr_qpos'] = curr_qpos
                        if hasattr(env, 'goal'):
                            frame_info['goal'] = env.goal
                        frame_info['contacts'] = env.sim.data.ncon
                        # env.visualize_dummy_indicator(env.sim.data.qpos[env.ref_joint_pos_indexes].copy())
                        self._store_frame(env, frame_info, planner=True)
                env.reset_color_agent()
            else:
                ll_ob = ob.copy()
                counter['rl'] += 1
                if not config.use_ik_target:
                    rescaled_ac = OrderedDict([('default', ac['default'].copy())])
                    if not config.discrete_action:
                        rescaled_ac['default'][:len(env.ref_joint_pos_indexes)] /=  config.omega
                    ob, reward, done, info = env.step(rescaled_ac)
                    contact_force = env.get_contact_force()
                    total_contact_force += contact_force
                else:
                    displacement['default'] /= config.omega
                    if 'gripper' in ac.keys():
                        displacement['default'] = np.concatenate((displacement['default'], ac['gripper']))
                    ob, reward, done, info = env.step(displacement)
                    contact_force = env.get_contact_force()
                    total_contact_force += contact_force
                ep_len += 1
                ep_rew += reward
                ep_rew_with_penalty += reward
                meta_len += 1
                meta_rew += reward
                reward_info.add(info)
                rollout.add({'done': done, 'rew': reward, 'rew_with_penalty': ep_rew_with_penalty})
                if record:
                    frame_info = info.copy()
                    frame_info['ac'] = ac['default']
                    frame_info['contact_force'] = contact_force
                    frame_info['std'] = np.array(stds['default'].detach().cpu())[0]
                    env.reset_visualized_indicator()
                    self._store_frame(env, frame_info)
            env._reset_prev_state()
            meta_rollout.add({'meta_done': done, 'meta_rew': meta_rew})
            env.reset_visualized_indicator()

        # last frame
        ll_ob = ob.copy()
        rollout.add({'ob': ll_ob, 'meta_ac': meta_ac})
        meta_rollout.add({'meta_ob': ob})

        ep_info.add({'len': ep_len, 'rew': ep_rew, "contact_force": total_contact_force, "avg_conntact_force": total_contact_force/ep_len})
        ep_info.add(counter)
        reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
        ep_info.add(reward_info_dict)
        # last frame
        return rollout.get(), meta_rollout.get(), ep_info.get_dict(only_scalar=True), self._record_frames

    def _store_frame(self, env, info={}, planner=False):
        color = (200, 200, 200)

        text = "{:4} {}".format(env._episode_length,
                                env._episode_reward)

        geom_colors = {}
        # if planner:
        #     for geom_idx in env.agent_geom_ids:
        #         color = env.sim.model.geom_rgba[geom_idx]
        #         geom_colors[geom_idx] = color.copy()
        #         color[0] = 0.0
        #         color[1] = 0.6
        #         color[2] = 0.4
        #         env.sim.model.geom_rgba[geom_idx] = color

        frame = env.render('rgb_array') * 255.0

        if self._config.vis_info:
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
