import env
import gym
from config import argparser
import numpy as np

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()
print(args)
print(args.env)

env = gym.make(args.env, **args.__dict__)
obs = env.reset()
print(len(obs['default']))

# def reset(env):
#     while True:
#         qpos = np.random.uniform(low=-1., high=1., size=env.model.nq) + env.sim.data.qpos.ravel()
#         qvel = np.random.uniform(low=-.005, high=.005, size=env.model.nv) + env.sim.data.qvel.ravel()
#         qvel[-2:] = 0
#         env.set_state(qpos, qvel)
#         env.sim.forward()
#         env.sim.step()
#         if env.sim.data.ncon == 0:
#             break
#     return env
#
# env = reset(env)
qpos = env.sim.data.qpos.ravel().copy()
qpos[0] = 1.75
qpos[1] = 0.879
qpos[2] = 0.87
qpos[3] = 0.701
qpos[4] = 0.589
qpos[5] = 1.55
qpos[6] = 1.39
env.set_state(qpos, env.sim.data.qvel.ravel())
env.sim.forward()
env.sim.step()

env.render(mode='human')
print("# contact: ", env.sim.data.ncon)
import pdb
pdb.set_trace()
env.step(np.ones(7)*0.0001)
print("# contact: ", env.sim.data.ncon)
env.render(mode='human')
import pdb
pdb.set_trace()
for i in range(1000):
    env.render(mode='human')
    action = env.action_space.sample()
    print(action)
    obs, reward, done, _ = env.step(action)
    if done:
        print('done')
        break
