import env
import gym
from config import argparser
import numpy as np
import time
import os
import cv2
import imageio
from util.misc import save_video, render_frame, mujocopy_render_hack
import torch
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video
from util.gym import render_frame, observation_size, action_size
from config.motion_planner import add_arguments as planner_add_arguments
import torchvision
from rl.sac_agent import SACAgent
from rl.policies import get_actor_critic_by_name
np.set_printoptions(precision=3)

mujocopy_render_hack() # rendering fix for gautam
is_save_video = False
parser = argparser()
config, unparsed = parser.parse_known_args()

from config.sawyer import add_arguments

add_arguments(parser)
planner_add_arguments(parser)
config, unparsed = parser.parse_known_args()

# config.camera_name = 'visview'
config.camera_name='cam0'
config.env = 'pusher-obstacle-hard-v3'
config.timelimit = 2.0

env = gym.make(config.env, **config.__dict__)
obs = env.reset()

config._xml_path = env.xml_path
config.device = torch.device("cpu")
config.is_chef = False
config.planner_integration = True
config.timelimit = 2.0

ob_space = env.observation_space
ac_space = env.action_space
joint_space = env.joint_space

allowed_collsion_pairs = []
geom_ids = env.agent_geom_ids + env.static_geom_ids
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
# env.render('human') # uncomment if you don't use mujocopy hack

frames = []
env.reset_visualized_indicator()
img = env.render('rgb_array')

curr_qpos = env.sim.data.qpos.copy()
target_qpos = curr_qpos.copy()
# targets = [np.array([-0.274, -0.846, 0.667, 0.254, -0.169, 0.0374, 0.0049]),
#     np.array([-0.274, -0.341, -0.0609, 0.67, 1.93, -2.26, 0.801]),
#             np.array([-0.335, -0.341, -0.0609, 0.852, 1.96, -1.87, 0.801])]

# targets  = [np.array([-1.01, -0.896, 0.304, 2.13, -0.0653, 0.0308, 0.00208])] # sawyer lift
# curr_qpos[env.ref_joint_pos_indexes] = targets[0]
# targets = [env.sim.data.qpos.copy()[env.ref_joint_pos_indexes]]
# targets  = [np.array([-0.153, -0.265, -0.00348, 1.28, 0.298, -0.298, 1.46])] # sawyer push

targets = [np.array([-1.16, -1.89, -1.2, -0.18])]
curr_qpos = env.sim.data.qpos.copy()
# curr_qpos[-4:-2] = np.array([-0.252, 0.16])
# env.set_state(curr_qpos, env.sim.data.qvel.ravel())

# env.set_state(curr_qpos, env.sim.data.qvel.copy())

is_target_vis = False
# target_qpos[env.ref_joint_pos_indexes] = np.array([-0.335, -0.341, -0.0609, 0.852, 1.96, -1.87, 0.801])

i = 0

# traj1 = np.load('traj_1.npy')
# traj2 = np.load('traj_2.npy')
# traj_list = [traj0, traj1, traj2]
# for target_id, traj in enumerate(traj_list):
#     for state in traj:t
#         env.visualize_goal_indicator(state[env.ref_joint_pos_indexes])
#         imageio.imsave('./tmp/vis/target_interm_{}_target_{}.png'.format(i, target_id),
#                                (env.render('rgb_array') * 255).astype(np.uint8))
#         i += 1

os.mkdir('tmp/vis/{}'.format(config.env))
for j, target in enumerate(targets):
    if is_target_vis:
        env.visualize_goal_indicator(target)
        imageio.imsave('./tmp/vis/{}_target_{}.png'.format(config.env, i),
                               (env.render('rgb_array') * 255).astype(np.uint8))
        i += 1
    else:
        target_qpos[env.ref_joint_pos_indexes] = target
        curr_qpos = env.sim.data.qpos.copy()
        # import pdb
        # pdb.set_trace()
        traj, _, _, _,_ = agent.plan(curr_qpos, target_qpos, ac_scale=env._ac_scale)
        # np.save('traj_{}.npy'.format(j), traj)

        for state in traj:
            imageio.imsave('./tmp/vis/{}/{}.png'.format(config.env,i),
                                   (env.render('rgb_array') * 255).astype(np.uint8))
            curr_qpos = env.sim.data.qpos.copy()
            curr_qpos[env.ref_joint_pos_indexes] = state[env.ref_joint_pos_indexes]
            env.set_state(curr_qpos, env.sim.data.qvel.ravel())
            i += 1
