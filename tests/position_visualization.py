import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config.robosuite import add_arguments
from config import argparser
from util.misc import make_ordered_pair, save_video
from config.motion_planner import add_arguments as planner_add_arguments
import cv2


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
add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

args.env = 'sawyer-lift-robosuite-v0'
args.camera_name = 'birdview'
env = gym.make(args.env, **args.__dict__)
mp_env = gym.make(args.env, **args.__dict__)
args._xml_path = env.xml_path


N = 1
frames = []
position = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
for episode, pos in enumerate(position):
    print("Episode: {}".format(episode))
    frames.append([])
    info = {}
    done = False
    ob = env.reset()

    current_qpos = env.sim.data.qpos.copy()
    current_qpos[env.ref_joint_pos_indexes] = np.zeros(len(env.ref_joint_pos_indexes))
    env.set_state(current_qpos, env.sim.data.qvel.ravel())
    for i in range(1):
        step = 0
        env.visualize_goal_indicator(np.zeros(len(env.ref_joint_pos_indexes)))

        action = np.ones(len(env.ref_joint_pos_indexes)) * -pos
        action = np.concatenate([action, [0.]])
        env.step(action)

        info['joint_diff'] = pos
        frames[episode].append(render_frame(env, step, info))

prefix_path = './tmp/visualization_test_sawyer/'
if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)
for i in range(len(position)):
    fpath = os.path.join(prefix_path, 'visualization_action_{}_bird.mp4'.format(position[i]))
    save_video(fpath, frames[i], fps=15)
