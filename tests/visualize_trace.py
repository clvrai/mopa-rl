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
args, unparsed = parser.parse_known_args()

from config.sawyer import add_arguments
args.env = 'sawyer-assembly-v0'

add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)
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
# env.render('human') # uncomment if you don't use mujocopy hack

frames = []
env.reset_visualized_indicator()
curr_qpos = env.sim.data.qpos.copy()
img = env.render('rgb_array')

target_qpos[env.ref_joint_pos_indexes] = np.array([-0.153, -0.164, -0.0304, 0.693, 1.93, -2.32, 0.848])
traj = agent.plan(target_qpos)

for i, state in enumerate(traj):
    imageio.imsave('{}.png'.format(i),
                           (env.render('rgb_array') * 255).astype(np.uint8))
    curr_qpos = env.sim.data.qpos.copy()
    curr_qpos[env.ref_joint_pos_indexes] = state
