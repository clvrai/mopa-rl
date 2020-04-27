import time
import argparse

import imageio
import numpy as np

from env import make_env


def argsparser():
    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser("Test env")
    parser.add_argument('--env', type=str, default='ant-reach')
    parser.add_argument('--render', type=str2bool, default=True)
    parser.add_argument('--save_img', type=str2bool, default=False)
    parser.add_argument('--sleep', type=int, default=0.01)
    parser.add_argument('--env_args', type=str, default=None)
    args = parser.parse_args()
    return args


def random_play_one_episode(env, render=False, save_img=False, sleep=0, random_len=0):
    if save_img:
        env.render_mode = 'rgb_array'
    if render:
        env.render_mode = 'human'

    ob = env.reset()
    total_rew = 0
    total_len = 0
    while True:
        if save_img:
            imageio.imsave('{}.png'.format(total_len),
                           (env.render('rgb_array') * 255).astype(np.uint8))

        if total_len < random_len:
            ob, rew, done, _ = env.step(env.action_space.sample())
        else:
            ob, rew, done, _ = env.step(np.zeros(env.action_space.sample().shape))

        if render:
            env.render()
            time.sleep(sleep)
        total_rew += rew
        total_len += 1
        if done:
            print('done')
            env.reset()
            if total_len < random_len:
                print("It's random action\'s fault")
            break
    print('Total reward: {}, Total length: {}'.format(total_rew, total_len))


def main(args):
    env = make_env(args.env, args)

    while True:
        random_play_one_episode(env, args.render, args.save_img, args.sleep)

    env.close()


if __name__ == '__main__':
    args = argsparser()
    main(args)

