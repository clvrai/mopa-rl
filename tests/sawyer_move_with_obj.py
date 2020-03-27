import os, sys
import numpy as np
import shutil

from motion_planners.sampling_based_planner import SamplingBasedPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose, qpos_from_site_pose_sampling
from config.motion_planner import add_arguments as planner_add_arguments
from robosuite.wrappers import IKWrapper
from robosuite.controllers import SawyerIKController
from math import pi
from util.misc import save_video
from util.gym import action_size, observation_size
import time
import cv2


parser = argparser()
args, unparsed = parser.parse_known_args()
from config.robosuite import add_arguments

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()


# Save video or not
is_save_video = True
record_caption = False

env = gym.make(args.env, **args.__dict__)
env.reset()
env.use_camera_obs = False
non_limited_idx = np.where(env._is_jnt_limited==0)[0]

print('robot_joints:  ', env.robot_joints)
print('gripper_names: ', env.gripper_joints)
print('num actions:   ', action_size(env.action_space))

state = env.sim.get_state()
print('qpos', env.sim.data.qpos[:])
print('state', state)
zero_state = state.qpos[:] * 0.0
state.qpos[:] = zero_state

env.sim.set_state(state)
print('zeroed state', env.sim.get_state())
des_qpos = np.copy(state.qpos)

print('control idx pos', env.ref_joint_pos_indexes)
print('control idx vel', env.ref_joint_vel_indexes)
print('gripper idx', env.ref_gripper_joint_pos_indexes)
print('# dof', env.dof)
print('# mujoco robot dof', env.mujoco_robot.dof)
print('dt', env.dt)
print('control_freq', env.control_freq)
print('control_timestep', env.control_timestep)
print('mujoco timestep', env.sim.model.opt.timestep)


outer_dt = env.control_timestep
inner_dt = env.sim.model.opt.timestep

n_inner_loop = int(outer_dt / inner_dt)
import ipdb; ipdb.set_trace()

# create a trajectory to goal
q_0 = env.sim.data.qpos[env.ref_joint_pos_indexes] # current pos
if 'sawyer-pick-move-robosuite-v0' in args.env:
    # set box gripping as goal position
    result = qpos_from_site_pose_sampling(env, 'grip_site', target_pos=env._get_pos('box'), target_quat=env._get_quat('box'), 
    joint_names=env.robot_joints, max_steps=100, tol=5e-2, trials=20, logging=True)
    print('IK optim for box pose successful? %s. Setting q_f to this IK result' % (result.success))
    q_f = result.qpos[env.ref_joint_pos_indexes]
else:
    q_f = np.zeros(env.mujoco_robot.dof)
    q_f -= 0.3

gripper_0 = env.sim.data.qpos[env.ref_gripper_joint_pos_indexes] # current pos
gripper_f = np.array([-0.01, 0.01])

# Create planner
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx)

N = 50
traj = np.linspace(q_0, q_f, N)
gripper_traj = np.linspace(gripper_0, gripper_f, N)
Kp = 300.0
Kd = 5.
Ki = 0.0 #5.0
alpha = 0.95

error = 0
x_traj_prev = traj[0, :]
i_term = np.zeros_like(x_traj_prev)
x_prev = env.sim.data.qpos[env.ref_joint_pos_indexes]
use_step = False

for t in range(1, N):
    x_traj = traj[t, :]
    xd_traj = (x_traj - x_traj_prev) / outer_dt

    gripper_action = gripper_traj[t, :]

    env.set_robot_indicator_joint_positions(x_traj)
    env.sim.data.qfrc_applied[env.ref_indicator_joint_pos_indexes] = env.sim.data.qfrc_bias[
        env.ref_indicator_joint_pos_indexes]

    if use_step:
        action = traj[t, :] - traj[t-1, :]
        action = np.append(action, gripper_action[0])
        env.step(action=action)

        env.render()
        time.sleep(outer_dt)
    else:
        for i in range(n_inner_loop):
            # gravity compensation
            env.sim.data.qfrc_applied[env.ref_joint_vel_indexes] = env.sim.data.qfrc_bias[env.ref_joint_vel_indexes]

            p_term = Kp * (x_traj - env.sim.data.qpos[env.ref_joint_pos_indexes])
            d_term = Kd * (xd_traj * 0 - env.sim.data.qvel[env.ref_joint_pos_indexes])
            i_term = alpha * i_term + Ki * (x_prev - env.sim.data.qpos[env.ref_joint_pos_indexes])

            print('p term ', np.linalg.norm(p_term))
            print('d term ', np.linalg.norm(d_term))
            print('i term ', np.linalg.norm(i_term))
            action = p_term + d_term + i_term

            # add gripper command to action
            # gripper_action = env.gripper.format_action(np.array([0.01]))
            action = np.concatenate([action, -gripper_action])

            env.sim.data.ctrl[:] = action[:]
            env.sim.forward()
            env.sim.step()

            env.render()
            time.sleep(inner_dt)

    print('x_traj', x_traj)
    print('x_t', env.sim.data.qpos[env.ref_joint_pos_indexes])
    print('gipper_action', gripper_action)
    print('gripper_state', env.sim.data.qpos[env.ref_gripper_joint_pos_indexes])

    x_prev = np.copy(env.sim.data.qpos[env.ref_joint_pos_indexes])
