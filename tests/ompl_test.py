import numpy as np

from motion_planners.sampling_based_planner import SamplingBasedPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose
from config.motion_planner import add_arguments as planner_add_arguments
from math import pi


parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make('reacher-obstacle-test-v0', **args.__dict__)
env.reset()

qpos = env.sim.data.qpos.ravel()
qvel = env.sim.data.qvel.ravel()

goal = env.sim.data.qpos[-2:]

ik_env = gym.make('reacher-obstacle-test-v0', **args.__dict__)
ik_env.reset()
ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
result = qpos_from_site_pose(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)
ik_env.close()

#start = np.concatenate([env.sim.data.qpos, env.sim.data.qvel])
#goal = np.concatenate([result.qpos, env.sim.data.qvel])
start = env.sim.data.qpos[:-2]
goal = result.qpos[:-2]
planner = SamplingBasedPlanner(args, env.xml_path, 7)
traj = planner.kinematic_plan(start, goal,  10., 0.1)


goal = env.sim.data.qpos[-2:]
print("Goal: ", goal)
print(traj)
for state in traj:
    env.render(mode='human')
    #env.step(state)
    env.set_state(np.concatenate((state, goal)).ravel(), env.sim.data.qvel.ravel())
env.render(mode='human')
import pdb
pdb.set_trace()
