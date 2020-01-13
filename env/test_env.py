import env
import gym

env = gym.make('reacher-obstacle-v0')
env.reset()

for i in range(100):
    env.render(mode='human')
    action = env.action_space.sample()
    env.step(action)

