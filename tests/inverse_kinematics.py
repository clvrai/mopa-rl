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
from util.env import quat_mul, mat2quat
from config.motion_planner import add_arguments as planner_add_arguments
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
import cv2
import time
import timeit
import copy
np.set_printoptions(precision=3)

mujocopy_render_hack() # workaround for mujoco py issue #390


parser = argparser()
config, unparsed = parser.parse_known_args()
target_site = 'fingertip'
if 'pusher' in config.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in config.env:
    from config.robosuite import add_arguments
    add_arguments(parser)
elif 'sawyer' in config.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
    target_site = 'grip_site'
elif 'reacher' in config.env:
    from config.reacher import add_arguments
    add_arguments(parser)



# Build Motion Planner ==============================
planner_add_arguments(parser)
config, unparsed = parser.parse_known_args()
env = gym.make(config.env, **config.__dict__)
config._xml_path = env.xml_path
config.device = torch.device("cpu")
config.is_chef = False
config.planner_integration = True
config.ik_target = 'grip_site'
config.action_range = 1.0

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


# ==============================================


# Reset IK Env (used in the invese kineamtics)
ik_env = gym.make(config.env, **config.__dict__)
ik_env.reset()

qpos = env.sim.data.qpos.ravel().copy()
qvel = env.sim.data.qvel.ravel().copy()
ik_env.set_state(qpos, qvel)


is_save_video = False
frames = []
done = False
ob = env.reset()
step = 0
if is_save_video:
    frames.append(render_frame(env, step))
else:
    env.render('human')

while True:
    curr_qpos = env.sim.data.qpos.copy()
    qpos = env.sim.data.qpos.ravel().copy()
    qvel = env.sim.data.qvel.ravel().copy()
    ik_env.set_state(qpos, qvel)
    env.set_state(qpos, qvel)

    # Actioon -- displacement of coordinates and orientation
    cart = np.random.uniform(low=[-0.2, -0.2, -0.2], high=[0.2, 0.2, 0.2])
    angle = np.random.uniform(low=-0.3, high=0.3)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis) * np.sin(angle/2.0)
    quat = np.array([np.cos(angle/2.0), axis[0], axis[1], axis[2]])  # (w,x,y,z)

    target_cart = np.clip(env.sim.data.get_site_xpos(target_site)[:len(env.min_world_size)] + config.action_range * cart, env.min_world_size, env.max_world_size)
    target_quat = mat2quat(env.sim.data.get_site_xmat(config.ik_target))  # (x,y,z,w)
    target_quat = target_quat[[3, 0, 1, 2]]  # (w,x,y,z)
    target_quat = quat_mul(target_quat, quat/np.linalg.norm(quat))

    print('current_cart', env.sim.data.get_site_xpos(config.ik_target)[:len(env.min_world_size)])
    print('target_cart', target_cart)
    print('target_quat', target_quat)
    print('angle [rad]', 2 * np.arccos(quat[0]))

    result = qpos_from_site_pose(ik_env, target_site, target_pos=target_cart,
                                 target_quat=target_quat, joint_names=env.robot_joints, max_steps=10, tol=1e-5)
    target_qpos = env.sim.data.qpos.copy()
    target_qpos[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes]

    env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())

    trial = 0
    while not agent.isValidState(target_qpos) and trial < 100:
        d = env.sim.data.qpos.copy()-target_qpos
        target_qpos += config.step_size * d/np.linalg.norm(d)
        trial += 1

    traj, success, interpolation, valid, exact = agent.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)
    print("Planning Statues: Exact {} Valid {}".format(exact, valid))
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
                frames.append(render_frame(env, step, info))
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
            frames.append(render_frame(env, step))
        else:
            import timeit
            t = timeit.default_timer()
            while timeit.default_timer() - t < 0.1:
                env.render('human')

if is_save_video:
    prefix_path = './tmp/inverse_kinematics_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    fpath = os.path.join(prefix_path, 'test.mp4')
    save_video(fpath, frames, fps=5)
