import numpy as np

from motion_planners.sampling_based_planner import SamplingBasedPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose


parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make('reacher-obstacle-v0', **args.__dict__)
env.reset()
env.render(mode='human')

goal = env.sim.data.qpos[-2:]

ik_env = gym.make('reacher-test-v0', **args.__dict__)
ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())

result = qpos_from_site_pose(env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)

start = np.concatenate([env.sim.data.qpos, env.sim.data.qvel])
goal = np.concatenate([result.qpos, env.sim.data.qvel])

planner = SamplingBasedPlanner(args, env.xml_path)
traj = planner.plan(start, goal, 1.0)

env.render(mode='human')
