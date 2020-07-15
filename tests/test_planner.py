import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
import torch
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video
from util.gym import render_frame, observation_size, action_size
from config.motion_planner import add_arguments as planner_add_arguments
import torchvision
from rl.sac_agent import SACAgent
from rl.policies import get_actor_critic_by_name
import time
import timeit
import copy
np.set_printoptions(precision=3)

parser = argparser()
config, unparsed = parser.parse_known_args()
if 'pusher' in config.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in config.env:
    from config.robosuite import add_arguments
    add_arguments(parser)
elif 'sawyer' in config.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
elif 'reacher' in config.env:
    from config.reacher import add_arguments
    add_arguments(parser)

planner_add_arguments(parser)
config, unparsed = parser.parse_known_args()
env = gym.make(config.env, **config.__dict__)
config._xml_path = env.xml_path
config.device = torch.device("cpu")
config.is_chef = False
config.planner_integration = True

ob_space = env.observation_space
ac_space = env.action_space
joint_space = env.joint_space

allowed_collsion_pairs = []
geom_ids = env.agent_geom_ids + env.static_geom_ids
if config.allow_manipulation_collision:
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in geom_ids:
            allowed_collsion_pairs.append(make_ordered_pair(manipulation_geom_id, geom_id))

ignored_contact_geom_ids = []
ignored_contact_geom_ids.extend(allowed_collsion_pairs)
config.ignored_contact_geom_ids = ignored_contact_geom_ids

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]
config.passive_joint_idx = passive_joint_idx

actor, critic = get_actor_critic_by_name(config.policy)

# build up networks
non_limited_idx = np.where(env.sim.model.jnt_limited[:action_size(env.action_space)]==0)[0]
agent = SACAgent(
    config, ob_space, ac_space, actor, critic, non_limited_idx, env.ref_joint_pos_indexes, env.joint_space, env._is_jnt_limited, env.jnt_indices
)


N = 1
is_save_video = True
frames = []

for episode in range(N):
    print("Episode: {}".format(episode))
    done = False
    ob = env.reset()
    curr_qpos = env.sim.data.qpos.copy()
    env.set_state(curr_qpos, env.sim.data.qvel)
    step = 0
    if is_save_video:
        frames.append([render_frame(env, step)])
    else:
        env.render('human')

    while not done:
        current_qpos = env.sim.data.qpos.copy()
        target_qpos = current_qpos.copy()
        # target_qpos[env.ref_joint_pos_indexes] += np.random.uniform(low=-2, high=2, size=len(env.ref_joint_pos_indexes))
        target_qpos[env.ref_joint_pos_indexes] = np.array([-0.732, -0.72, 0.0304, 1.77, -0.179, 0, 0.283])

        trial = 0
        while not agent.isValidState(target_qpos) and trial < 100:
            d = env.sim.data.qpos.copy()-target_qpos
            target_qpos += config.step_size * d/np.linalg.norm(d)
            trial+=1

        traj, success, interpolation, valid, exact = agent.plan(current_qpos, target_qpos, ac_scale=env._ac_scale)
        env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())

        reward = 0
        if success:
            for j, next_qpos in enumerate(traj):
                action = env.form_action(next_qpos)
                env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())
                ob, reward, done, info = env.step(action, is_planner=True)
                step += 1
                if is_save_video:
                    info['ac'] = action['default']
                    info['next_qpos'] = next_qpos
                    info['target_qpos'] = target_qpos
                    info['curr_qpos'] = env.sim.data.qpos.copy()
                    frames[episode].append(render_frame(env, step, info))
                else:
                    import timeit
                    t = timeit.default_timer()
                    while timeit.default_timer() - t < 0.1:
                        env.render('human')
                if done or step > config.max_episode_steps:
                    break
        else:
            step += 1
            if step > config.max_episode_steps:
                break
            if is_save_video:
                frames[episode].append(render_frame(env, step))
            else:
                import timeit
                t = timeit.default_timer()
                while timeit.default_timer() - t < 0.1:
                    env.render('human')


if is_save_video:
    prefix_path = './tmp/motion_planning_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    for i, episode_frames in enumerate(frames):
        fpath = os.path.join(prefix_path, 'planner_stest_{}_{}.mp4'.format(config.env, i))
        save_video(fpath, episode_frames, fps=5)
