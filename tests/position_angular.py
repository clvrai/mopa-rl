import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
# from config.pusher import add_arguments
from config.robosuite import add_arguments
from config import argparser
from util.misc import make_ordered_pair, save_video
from config.motion_planner import add_arguments as planner_add_arguments
import matplotlib.pyplot as plt
import cv2


parser = argparser()
add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

args.env = 'sawyer-lift-robosuite-v0'
env = gym.make(args.env, **args.__dict__)
mp_env = gym.make(args.env, **args.__dict__)
args._xml_path = env.xml_path


N = 1
frames = []
position = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
action_scales = np.linspace(0, 2, 19)
data = [[] for _ in range(len(env.ref_joint_pos_indexes))]
for i, scale in enumerate(action_scales):
    print(i)
    ob = env.reset()
    current_qpos = env.sim.data.qpos.copy()
    current_qpos[env.ref_joint_pos_indexes] = np.zeros(len(env.ref_joint_pos_indexes))
    env.set_state(current_qpos, env.sim.data.qvel.ravel())

    action = np.ones(env.dof) * scale
    env.step(action)
    qpos = env.sim.data.qpos.ravel().copy()
    for j, pos in enumerate(qpos[env.ref_joint_pos_indexes]):
        data[j].append(pos)

data = np.array(data)
for k in range(len(env.ref_joint_pos_indexes)):
    plt.figure()
    plt.plot(action_scales, data[k])
    plt.title("Radian vs Action: Joint: {}".format(k))
    plt.ylabel("Radian")
    plt.xlabel("Action")
    plt.show()
