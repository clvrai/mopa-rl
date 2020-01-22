import os, sys

import env
import gym
from config import argparser
from motion_planners.komo import KOMO
from env.inverse_kinematics import qpos_from_site_pose
from copy import deepcopy
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

goal = env.sim.data.qpos[-2:]

mp = KOMO('./env/assets/rai/reacher.g', False)
traj = mp.plan_motion(goal, qpos[:-2])

ik_env = gym.make('reacher-test-v0', **args.__dict__)
ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())


env.render(mode='human')
result = qpos_from_site_pose(env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)
env.render(mode='human')
print(result)
import pdb
pdb.set_trace()

#result = qpos_from_site_pose(env, 'fingertip', target_pos=env._get_pos('target'), target_quat=None, joint_names=env.model.joint_names[:-2], max_steps=300)
#env.step(result.qpos-env.sim.data.qpos.ravel()[:-2])
#env.render(mode='human')

#frames = []

#for i, pos in enumerate(traj):
#    env.render(mode='human')
#    print(pos)
#    action = pos - env.sim.data.qpos.ravel()[:-2]
#    print(action)
    #ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    #result = qpos_from_site_pose(ik_env, 'fingertip', target_pos=pos[:3], target_quat=pos[3:], joint_names=env.model.joint_names[:-2], max_steps=300)
    #frame = env.render(mode='rgb_array') * 255.
    #frames.append(frame)
#    env.step(action)
#    print(env.sim.data.qpos.ravel()[:-2])
#    print('=============================')
    #qpos[:-2] = pos
    #env.set_state(qpos, qvel)

#frames = np.stack(frames)
#path = './test_logs/motion_planner/test_0.mp4'
#save_video(path, frames)

#
