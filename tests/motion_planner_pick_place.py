import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
import torch
from rl.planner_agent import PlannerAgent
import torchvision
from rl.sac_agent import SACAgent
from rl.policies import get_actor_critic_by_name
from util.misc import make_ordered_pair, save_video, mujocopy_render_hack
from util.gym import render_frame, observation_size, action_size
from config.motion_planner import add_arguments as planner_add_arguments
from env.inverse_kinematics import qpos_from_site_pose_sampling
import cv2
import time
import timeit
import copy
np.set_printoptions(precision=3)

mujocopy_render_hack() # workaround for mujoco py issue #390


def get_goal_position(env, goal_site='target', z_offset = 0.): # use goal_site='cube' for pick-place-v0
    ik_env = gym.make(config.env, **config.__dict__)
    ik_env.reset()

    qpos = env.sim.data.qpos.ravel().copy()
    qvel = env.sim.data.qvel.ravel().copy()
    ik_env.set_state(qpos, qvel)

    # Obtain goal joint positions. Do IK to get joint positions for goal_site.
    # target quat set to picking from above [0, 0, 1, 0]
    # result = qpos_from_site_pose_sampling(ik_env, 'grip_site', target_pos=(env._get_pos(goal_site) + np.array([0., 0., z_offset])),
    #             target_quat=np.array([0., 0., 1., 0.]), joint_names=env.robot_joints, max_steps=1000, tol=1e-3)
    result = qpos_from_site_pose_sampling(ik_env, 'grip_site', target_pos=(env._get_pos(goal_site) + np.array([0., 0., z_offset])),
                target_quat=np.array([0., 0., 1., 0.]), joint_names=env.robot_joints, max_steps=1000, tol=1e-3)

    print("IK for %s successful? %s. Err_norm %.5f" % (goal_site, result.success, result.err_norm))
    # Equate qpos components not affected by planner
    goal = qpos.copy()
    goal[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes]
    ik_env.set_state(goal, qvel)
    # print(goal[env.ref_joint_pos_indexes])
    # ik_env.render('human')
    # input("See if IK solution is fine. Press any key to continue; Ctrl-C to quit")

    return goal

parser = argparser()
config, unparsed = parser.parse_known_args()
if 'pusher' in config.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in config.env:
    from config.robosuite import add_arguments
    add_arguments(parser)
elif 'sawyer' in config.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
elif 'reacher' in config.env:
    from config.reacher import add_arguments
    add_arguments(parser)

planner_add_arguments(parser)
config, unparsed = parser.parse_known_args()
config.camera_name = 'frontview'
env = gym.make(config.env, **config.__dict__)
config._xml_path = env.xml_path
config.device = torch.device("cpu")
config.is_chef = False
config.planner_integration = True

ob_space = env.observation_space
ac_space = env.action_space
joint_space = env.joint_space

allowed_collsion_pairs = []
geom_ids = env.agent_geom_ids + env.static_geom_ids
if config.allow_manipulation_collision:
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in geom_ids:
            allowed_collsion_pairs.append(make_ordered_pair(manipulation_geom_id, geom_id))

ignored_contact_geom_ids = []
ignored_contact_geom_ids.extend(allowed_collsion_pairs)
config.ignored_contact_geom_ids = ignored_contact_geom_ids # contacts to ignore for planner

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]
config.passive_joint_idx = passive_joint_idx # joints not articulated by robot

actor, critic = get_actor_critic_by_name(config.policy)

# build up networks
non_limited_idx = np.where(env.sim.model.jnt_limited[:action_size(env.action_space)]==0)[0]
agent = SACAgent(
    config, ob_space, ac_space, actor, critic, non_limited_idx, env.ref_joint_pos_indexes, env.joint_space, env._is_jnt_limited, env.jnt_indices
)

N = 1
is_save_video = True # cannot do env.render('human') if you are saving video (env.render('rgb_array'))
frames = []
for episode in range(N):
    print("Episode: {}".format(episode))
    done = False
    ob = env.reset()
    curr_qpos = env.sim.data.qpos.copy()
    curr_qpos[env.ref_gripper_joint_pos_indexes] = -0.008
    env.set_state(curr_qpos, env.sim.data.qvel.ravel())
    curr_qpos = env.sim.data.qpos.copy()
    step = 0
    if is_save_video:
        frames.append([render_frame(env, step)])
    else:
        env.render('human')

    # First move above
    goal_joint_pos = get_goal_position(env, goal_site='cube', z_offset=0.1)
    optional_place_target = goal_joint_pos
    target_qpos = curr_qpos.copy()
    target_qpos[env.ref_joint_pos_indexes] = goal_joint_pos[env.ref_joint_pos_indexes]
    print("Goal %s" % target_qpos)
    print("Cube pos %s\t quat%s" % (env._get_pos('cube'), env._get_quat('cube')))

    env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
    trial = 0
    while not agent.isValidState(target_qpos) and trial < 100:
        d = env.sim.data.qpos.copy()-target_qpos
        target_qpos += config.step_size * d/np.linalg.norm(d)
        trial+=1

    traj, success, interpolation, valid, exact = agent.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)

    rewards = 0
    if success:
        for j, next_qpos in enumerate(traj):
            action = env.form_action(next_qpos)
            env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())
            action['default'][-1] = -1.0
            ob, reward, done, info = env.step(action, is_planner=True)
            step += 1
            if is_save_video:
                info['ac'] = action['default']
                info['next_qpos'] = next_qpos
                info['target_qpos'] = target_qpos
                info['curr_qpos'] = env.sim.data.qpos.copy()
                info['reward'] = reward
                frames[episode].append(render_frame(env, step, info))
            else:
                import timeit
                t = timeit.default_timer()
                while timeit.default_timer() - t < 0.1:
                    env.render('human')
            if done or step > config.max_episode_steps:
                break
    else:
        step += 1
        if step > config.max_episode_steps:
            break
        if is_save_video:
            frames[episode].append(render_frame(env, step))
        else:
            import timeit
            t = timeit.default_timer()
            while timeit.default_timer() - t < 0.1:
                env.render('human')

    # Then move down to cube
    goal_joint_pos = get_goal_position(env, goal_site='cube', z_offset=0.0)
    curr_qpos = env.sim.data.qpos.copy()
    target_qpos = curr_qpos.copy()
    target_qpos[env.ref_joint_pos_indexes] = goal_joint_pos[env.ref_joint_pos_indexes]
    # target_qpos[env.ref_gripper_joint_pos_indexes] = 0.15
    print("Goal %s" % target_qpos)
    print("Cube pos %s\t quat%s" % (env._get_pos('cube'), env._get_quat('cube')))

    env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
    trial = 0
    while not agent.isValidState(target_qpos) and trial < 100:
        d = env.sim.data.qpos.copy()-target_qpos
        target_qpos += config.step_size * d/np.linalg.norm(d)
        trial+=1

    traj, success, interpolation, valid, exact = agent.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)

    rewards = 0
    if success:
        for j, next_qpos in enumerate(traj):
            action = env.form_action(next_qpos)
            env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())
            action['default'][-1] = -1.0
            ob, reward, done, info = env.step(action, is_planner=True)
            step += 1
            if is_save_video:
                info['ac'] = action['default']
                info['next_qpos'] = next_qpos
                info['target_qpos'] = target_qpos
                info['curr_qpos'] = env.sim.data.qpos.copy()
                info['reward'] = reward
                frames[episode].append(render_frame(env, step, info))
            else:
                import timeit
                t = timeit.default_timer()
                while timeit.default_timer() - t < 0.1:
                    env.render('human')
            if done or step > config.max_episode_steps:
                break
    else:
        step += 1
        if step > config.max_episode_steps:
            break
        if is_save_video:
            frames[episode].append(render_frame(env, step))
        else:
            import timeit
            t = timeit.default_timer()
            while timeit.default_timer() - t < 0.1:
                env.render('human')

    # Then close gripper
    close_gripper_action = OrderedDict([('default', np.array([0, 0, 0, 0, 0, 0, 0, 1.0]))])
    for temp in range(10):
        ob, reward, done, info = env.step(close_gripper_action, is_planner=False)
        if is_save_video:
            frames[episode].append(render_frame(env, step))
        else:
            import timeit
            t = timeit.default_timer()
            while timeit.default_timer() - t < 0.1:
                env.render('human')

    while True:
        # Then move to goal position
        curr_qpos = env.sim.data.qpos.copy()
        target_qpos = curr_qpos.copy()
        # target_qpos[env.ref_joint_pos_indexes] = optional_place_target[env.ref_joint_pos_indexes]
        target_qpos[env.ref_joint_pos_indexes] += np.random.uniform(low=-1, high=1, size=len(env.ref_joint_pos_indexes))

        env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
        trial = 0
        while not agent.isValidState(target_qpos) and trial < 100:
            d = env.sim.data.qpos.copy()-target_qpos
            target_qpos += config.step_size * d/np.linalg.norm(d)
            trial+=1

        traj, success, interpolation, valid, exact = agent.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)

        rewards = 0
        print("Success?: ", success)
        if success:
            for j, next_qpos in enumerate(traj):
                action = env.form_action(next_qpos)
                env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())
                action['default'][-1] = 1.0
                ob, reward, done, info = env.step(action, is_planner=True)
                step += 1
                if is_save_video:
                    info['ac'] = action['default']
                    info['next_qpos'] = next_qpos
                    info['target_qpos'] = target_qpos
                    info['curr_qpos'] = env.sim.data.qpos.copy()
                    info['reward'] = reward
                    frames[episode].append(render_frame(env, step, info))
                else:
                    import timeit
                    t = timeit.default_timer()
                    while timeit.default_timer() - t < 0.1:
                        env.render('human')
                if done or step > config.max_episode_steps:
                    break
            if done or step > config.max_episode_steps:
                break
        else:
            step += 1
            if step > config.max_episode_steps:
                break
            if is_save_video:
                frames[episode].append(render_frame(env, step))
            else:
                import timeit
                t = timeit.default_timer()
                while timeit.default_timer() - t < 0.1:
                    env.render('human')


if is_save_video:
    prefix_path = './tmp/motion_planning_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    for i, episode_frames in enumerate(frames):
        fpath = os.path.join(prefix_path, 'test_trial_{}.mp4'.format(i))
        save_video(fpath, episode_frames, fps=5)
