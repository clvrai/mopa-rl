import os, sys
import numpy as np
import shutil

import gym
import env
from config.pusher import add_arguments
from config import argparser
from rl.planner_agent import PlannerAgent

parser = argparser()
add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make('simple-pusher-obstacle-v0', **args.__dict__)

passive_joint_idx = []
ignored_contacts = []
non_limited_idx = np.where(env._is_jnt_limited==0)[0]
# planenr = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts)

