import os, sys
import numpy as np
import shutil

from motion_planners.sampling_based_planner import SamplingBasedPlanner, SamplingBasedKinodynamicPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose, qpos_from_site_pose_sampling
from config.motion_planner import add_arguments as planner_add_arguments
from math import pi
from util.misc import save_video
from util.gym import action_size
import time

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

is_save_video = False

env = gym.make(args.env, **args.__dict__)
env_prime = gym.make(args.env, **args.__dict__)
ik_env = gym.make(args.env, **args.__dict__)
non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = SamplingBasedKinodynamicPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx)

start_time = time.time()
env.reset()
qpos = env.sim.data.qpos.ravel()
qvel = env.sim.data.qvel.ravel()
goal = env.goal
print("Initial Joint pos: ", qpos)
print("Initial Vel: ", qvel)
print("Goal: ", goal)

ik_env.reset()
ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
env_prime.reset()
env_prime.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
#result = qpos_from_site_pose(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)
result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=100)
ik_env.close()

start = env.sim.data.qpos.ravel()
goal = result.qpos


actions = planner.plan(np.concatenate((start, env.sim.data.qvel.ravel())), np.concatenate((goal, env.sim.data.qvel.ravel())),  args.timelimit)


goal = env.sim.data.qpos[-2:]
frames = []
action_frames = []
# for state in traj:
#     if is_save_video:
#         frame = env.render(mode='rgb_array')
#         frames.append(frame*255.)
#     else:
#         env.render(mode='human')
#     env.set_state(np.concatenate((state[:-2], goal)).ravel(), env.sim.data.qvel.ravel())
#
# if is_save_video:
#     frame = env.render(mode='rgb_array')
#     frames.append(frame*255.)
#     prefix_path = os.path.join('./tmp', args.planner_type)
#     if not os.path.exists(prefix_path):
#         os.mkdir(prefix_path)
#     fpath = os.path.join(prefix_path, 'action_{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
#     save_video(fpath, frames)
# else:
#     env.render(mode='human')
#

for action in actions:
    if is_save_video:
        frame = env_prime.render(mode='rgb_array')
        action_frames.append(frame*255.)
    else:
        env_prime.render(mode='human')
    env_prime.step(action)
if is_save_video:
    frame = env_prime.render(mode='rgb_array')
    action_frames.append(frame*255.)
    prefix_path = os.path.join('./tmp', args.planner_type)
    if not os.path.exists(prefix_path):
        os.mkdir(prefix_path)
    fpath = os.path.join(prefix_path, 'action_{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
    save_video(fpath, action_frames)
else:
    env_prime.render(mode='human')


