import os, sys

import env
import gym
from config import argparser
from rl.motion_planner import MotionPlanner
import numpy as np
import moviepy.editor as mpy

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


def save_video(path, frames, fps=1.):
    def f(t):
        frame_length = len(frames)
        new_fps = 1./(1./fps + 1./frame_length)
        idx = min(int(t*new_fps), frame_length-1)
        return frames[idx]

    video = mpy.VideoClip(f, duration=len(frames)/fps+2)

    video.write_videofile(path, fps, verbose=False)

add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make('reacher-test-v0', **args.__dict__)
env.reset()
qpos = env.sim.data.qpos
qvel = env.sim.data.qvel

#env.set_state(np.array([-0.00697466, 0.06072485,  0.00776119,  0.05877283, -0.04347407,  0.03286544,
#  0.04532825,  0.02162226, -0.19511766]), qvel)

print(env.sim.data.qpos)
goal = env.sim.data.qpos[-2:]
print(goal)

mp = MotionPlanner('./env/assets/rai/reacher_obstacle.g', False)
traj = mp.plan_motion(goal, qpos[:-2])

frames = []
for i, pos in enumerate(traj):
    print('i=', i, ': ', pos)
    #env.render(mode='human')
    frame = env.render(mode='rgb_array') * 255.
    frames.append(frame)
    qpos[:-2] = pos
    env.set_state(qpos, qvel)

frames = np.stack(frames)
path = './test_logs/motion_planner/test_0.mp4'
save_video(path, frames)


