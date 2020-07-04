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
    # action = np.ones(env.dof)
    # action = np.zeros(env.dof)
    # action[0] = -0.5
    # action[1] = -0.5
    # qpos = env.sim.data.qpos.ravel().copy()[env.ref_joint_pos_indexes].copy() + action['default'][:env.mujoco_robot.dof]
    # env.set_robot_indicator_joint_positions(qpos)
    # print("Action:\t%s" % action)
    obs, reward, done, _ = env.step(action)
    if is_save_video:
        # WARNING: don't is_save_video with a 'while True` loop since it runs forever
        frames[0].append(render_frame(env, i))
    else:
        env.render(mode='human')

    if done:
        print('done')
        break

if is_save_video:
    prefix_path = './tmp/motion_planning_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    # for i, episode_frames in enumerate(frames):
    fpath = os.path.join(prefix_path, 'test_trial.mp4')
    save_video(fpath, frames[0], fps=5)
