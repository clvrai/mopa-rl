import env
import gym
from config import argparser
import numpy as np

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'mover' in args.env:
    from config.mover import add_arguments
elif 'peg-insertion' in args.env:
    from config.peg_insertion import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments

else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)
if 'robosuite' in args.env:
    env.use_camera_obs = False
obs = env.reset()

#for i in range(10000):
timestep = 0
while True:
    env.render(mode='human')
    action = env.action_space.sample()
    qpos = env.sim.data.qpos.ravel().copy()[env.ref_joint_pos_indexes] + action['default'][:-1]
    env.set_robot_indicator_joint_positions(qpos)
    obs, reward, done, _ = env.step(action)
    print(timestep)
    if done:
        print('done')
        break
