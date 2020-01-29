import os, sys
import numpy as np
import shutil

from motion_planners.sampling_based_planner import SamplingBasedPlanner
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

is_save_video = True

env = gym.make(args.env, **args.__dict__)
env_prime = gym.make(args.env, **args.__dict__)
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space))

def run_mp(env, planner, i=None):
    error = 0
    env.reset()
    qpos = env.sim.data.qpos.ravel()
    qvel = env.sim.data.qvel.ravel()
    #env.goal = [-0.25128643, 0.14829235]
    #qpos[-2:] = env.goal
    #qpos[:-2] = [0.09539838, 0.04237122, 0.05476331, -0.0676346, -0.0434791, -0.06203809, 0.03571644]
    #qvel[:-2] = [ 0.00293847, 0.00158573, 0.0018593, 0.00122192, -0.0016253, 0.00225007, 0.00001702]
    env.set_state(qpos, qvel)
    goal = env.goal


    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()
    ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    env_prime.reset()
    env_prime.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    #result = qpos_from_site_pose(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)
    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=100)
    ik_env.close()

    start = env.sim.data.qpos.ravel()
    goal = result.qpos

    traj, actions = planner.plan(start, goal,  args.timelimit, is_simplified=True)


    goal = env.sim.data.qpos[-2:]
    frames = []
    action_frames = []
    for state in traj[1:]:
        if is_save_video:
            frame = env.render(mode='rgb_array')
            frames.append(frame*255.)
        else:
            env.render(mode='human')
        #env.set_state(np.concatenate((state[:-2], goal)).ravel(), env.sim.data.qvel.ravel())
        env.step(-(env.sim.data.qpos[:-2]-state[:-2])*env._frame_skip)
        error += np.sqrt((env.sim.data.qpos - state)**2)

    if is_save_video:
        frame = env.render(mode='rgb_array')
        frames.append(frame*255.)
        prefix_path = os.path.join('./tmp', args.planner_type)
        if not os.path.exists(prefix_path):
            os.mkdir(prefix_path)
        if i is None:
            i == ""
        fpath = os.path.join(prefix_path, 'action_{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
        save_video(fpath, frames, fps=5)
    else:
        env.render(mode='human')


    return error / len(traj[1:])

errors = 0
N = 10
for i in range(N):
    error = run_mp(env, planner, i)
    errors += error

print(errors/N)

# for action in actions:
#     if is_save_video:
#         frame = env_prime.render(mode='rgb_array')
#         action_frames.append(frame*255.)
#     else:
#         env_prime.render(mode='human')
#     env_prime.step(action)
#
# if is_save_video:
#     frame = env_prime.render(mode='rgb_array')
#     action_frames.append(frame*255.)
#     prefix_path = os.path.join('./tmp', args.planner_type)
#     if not os.path.exists(prefix_path):
#         os.mkdir(prefix_path)
#     fpath = os.path.join(prefix_path, 'action_{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
#     save_video(fpath, action_frames)
# else:
#     env_prime.render(mode='human')
#
#
