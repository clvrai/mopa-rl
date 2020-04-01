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
import cv2
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'mover' in args.env:
    from config.mover import add_arguments
else:
    raise ValueError('args.env (%s) is not supported for this test script' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

# Save video or not
is_save_video = True
record_caption = True

env = gym.make('simple-mover-v0', **args.__dict__)
env.reset()
env.render(mode='human')
env_test = gym.make('simple-mover-test-v0', **args.__dict__)
env_test.reset()
qpos = env_test.sim.data.qpos.ravel().copy()
qpos[:3] = env.sim.data.qpos.ravel().copy()[:3]
env_test.set_state(qpos, env_test.sim.data.qvel.ravel().copy())
body_id = env_test.sim.model.body_name2id('box')
env_test.sim.model.body_pos[body_id] = (env.sim.data.get_body_xpos('box'))
env_test.sim.forward()
env_test.render('human')
import pdb
pdb.set_trace()
