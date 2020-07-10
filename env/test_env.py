import env
import gym
from config import argparser
import numpy as np
import time
import os
import cv2
from util.misc import save_video, render_frame, mujocopy_render_hack
np.set_printoptions(precision=3)

mujocopy_render_hack() # rendering fix for gautam
is_save_video = False
parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'mover' in args.env:
    from config.mover import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
elif 'jaco' in args.env:
    from config.jaco import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)
if 'robosuite' in args.env:
    env.use_camera_obs = False
obs = env.reset()
obs = env.reset()
# env.render('human') # uncomment if you don't use mujocopy hack

frames = []
frames.append([render_frame(env, 0)])
env.reset_visualized_indicator()
for i in range(100):
# while True:
    # env.render(mode='rgb_array')
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if is_save_video:
        # WARNING: don't is_save_video with a 'while True` loop since it runs forever
        frames[0].append(render_frame(env, i))
    else:
        env.render(mode='human')
    if done:
        print('done')
        break

