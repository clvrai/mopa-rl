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
from mujoco_py import MjSim, MjViewer, functions

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


def inverse_dynamics(env):
    partialM = env.sim.data.qM
    M = np.zeros(env.model.nq*env.model.nq)
    L = functions.mj_fullM(env.sim.model, M, env.sim.data.qM)
    M = M.reshape((env.model.nq, env.model.nq))
    acc = env.sim.data.qacc
    c = env.sim.data.qfrc_bias
    J = env.sim.data.efc_J
    f = env.sim.data.efc_force
    return np.matmul(M, acc) + c - np.matmul(J.T, f)

add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

is_save_video = False

env = gym.make(args.env, **args.__dict__)
env_dynamics = gym.make(args.env, **args.__dict__)
ik_env = gym.make(args.env, **args.__dict__)
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space))

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
env_dynamics.reset()
env_dynamics.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())

result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=100)
ik_env.close()

start = env.sim.data.qpos.ravel()
goal = result.qpos


traj, actions = planner.plan(start, goal,  args.timelimit)


goal = env.sim.data.qpos[-2:]
frames = []
action_frames = []

for state in traj:
    env_dynamics.render(mode='human')
    env.set_state(np.concatenate((state[:-2], goal)).ravel(), env.sim.data.qvel.ravel())
    action = inverse_dynamics(env)
    env_dynamics.step(action[:-2])
import pdb
pdb.set_trace()




