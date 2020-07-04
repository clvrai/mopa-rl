import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video
from config.motion_planner import add_arguments as planner_add_arguments
import cv2
import time
import timeit
import copy
np.set_printoptions(precision=3)

def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    offset = 12
    x, y = 5, fheight+10
    cv2.putText(frame, text,
                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, (255, 255, 0), thickness, cv2.LINE_AA)

    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = '{}: '.format(k)
        (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_size, thickness)
        cv2.putText(frame, key_text,
                    (x, y+offset*(i+2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (66, 133, 244), thickness, cv2.LINE_AA)
        cv2.putText(frame, str(v),
                    (x + key_width, y+offset*(i+2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

parser = argparser()
args, unparsed = parser.parse_known_args()
if 'pusher' in args.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
elif 'reacher' in args.env:
    from config.reacher import add_arguments
    add_arguments(parser)
else:
    raise NotImplementedError

planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)

max_actions = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
min_actions = [-0.1, -0.3, -0.5, -1.0, -1.5, -2.0]

frames = []
for i, (max_action, min_action) in enumerate(zip(max_actions, min_actions)):
    curr_state = env.sim.data.qpos.copy()
    curr_state[env.ref_joint_pos_indexes] = np.zeros(len(env.ref_joint_pos_indexes))
    env.set_state(curr_state, env.sim.data.qvel.ravel())

    frames.append([])
    for step in range(100):
        curr_state[env.ref_joint_pos_indexes] = np.zeros(len(env.ref_joint_pos_indexes))
        env.set_state(curr_state, env.sim.data.qvel.ravel())

        action = np.random.uniform(low=min_action, high=max_action, size=env.dof)
        curr_state = env.sim.data.qpos.copy()
        target_state = curr_state.copy()
        target_state[env.ref_joint_pos_indexes] = curr_state[env.ref_joint_pos_indexes] + action[:len(env.ref_joint_pos_indexes)]
        env.set_state(target_state, env.sim.data.qvel.ravel())

        info = dict(action=action)
        frames[i].append(render_frame(env, step, info))

prefix_path = './tmp/action_range/'
if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)
for i, episode_frames in enumerate(frames):
    fpath = os.path.join(prefix_path, 'ac_max_{}.mp4'.format(max_actions[i]))
    save_video(fpath, episode_frames, fps=5)
