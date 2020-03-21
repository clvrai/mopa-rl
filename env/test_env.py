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
elif 'sawyer-move' in args.env:
    from config.sawyer_move import add_arguments
elif 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'mover' in args.env:
    from config.mover import add_arguments
elif 'pick-place' in args.env:
    from config.sawyer_pick_place import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments

else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make(args.env, **args.__dict__)
obs = env.reset()

#for i in range(10000):
while True:
    env.render(mode='human')
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    # if done:
    #     print('done')
    #     break
