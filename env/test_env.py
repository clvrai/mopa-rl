import env
import gym
from config import argparser
import numpy as np
import time
import os
import cv2
from util.misc import save_video, render_frame, mujocopy_render_hack
np.set_printoptions(precision=3)

mujocopy_render_hack() # rendering fix
is_save_video = False
parser = argparser()
args, unparsed = parser.parse_known_args()

if 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)

obs = env.reset()

env.reset_visualized_indicator()
curr_qpos = env.sim.data.qpos.copy()
# curr_qpos[:4] = np.zeros(4)
# env.set_state(curr_qpos, env.sim.data.qvel.copy())
while True:
    # env.render(mode='rgb_array')
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print('done')
        break

