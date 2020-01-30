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
qpos = env.sim.data.qpos.ravel()
qvel = env.sim.data.qvel.ravel()

env.goal = [-0.25128643, 0.14829235]
qpos[-2:] = env.goal
qpos[:-2] = [0.09539838, 0.04237122, 0.05476331, -0.0676346, -0.0434791, -0.06203809, 0.03571644]
qvel[:-2] = [ 0.00293847, 0.00158573, 0.0018593, 0.00122192, -0.0016253, 0.00225007, 0.00001702]


env._set_pos('subgoal1', [0.2, 0.05,0])
env._set_pos('subgoal2', [0.03, 0.1,0])
env._set_pos('subgoal3', [-0.18, 0.05,0])

env.set_state(qpos, qvel)

for i in range(200):
    env.render(mode='human')
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        print('done')
        break
