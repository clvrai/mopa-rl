import env
import gym
import os
from config import argparser
import numpy as np

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'sawyer-move' in args.env:
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

fname = args.env.replace("-v0", "")
fname = fname.replace("-", "_")
env.model.save_model(os.path.join('./env/assets/xml/', fname+'.xml'))

