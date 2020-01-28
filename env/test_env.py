import env
import gym
from config import argparser

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make(args.env, **args.__dict__)
env.reset()

for i in range(200):
    env.render(mode='human')
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        print('done')
        break
