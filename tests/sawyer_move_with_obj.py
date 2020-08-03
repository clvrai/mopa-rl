import os, sys
import numpy as np
import shutil
from util.logger import logger
from motion_planners.sampling_based_planner import SamplingBasedPlanner
import env
import gym
from config import argparser
from env.inverse_kinematics import qpos_from_site_pose, qpos_from_site_pose_sampling
from config.motion_planner import add_arguments as planner_add_arguments
from math import pi
from util.misc import save_video
from util.gym import action_size
import time
import cv2
from util.contact_info import print_contact_info
# workaround for mujoco py issue #390
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

def make_ordered_pair(id1, id2):
    return (min(id1, id2), max(id1, id2))

def check_state_validity(env, ignored_contacts):
    valid = True # assume valid state; all current contacts are to be ignored
    if env.sim.data.ncon > 0:
        for i in range(env.sim.data.ncon):
            con_data = env.sim.data.contact[i]
            con_pair = make_ordered_pair(con_data.geom1, con_data.geom2)
            try:
                ignored_contacts.index(con_pair)
            except ValueError:
                logger.info("Contact pair (%d, %d) not in list of contacts to ignore. Invalid state." % (con_pair[0], con_pair[1]))
                valid = False
                break
    return valid

def render_frame(env, step, info={}):
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    if record_caption:
        font_size = 0.4
        thickness = 1
        offset = 12
        x, y = 5, fheight+10
        cv2.putText(frame, text,
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 0), thickness, cv2.LINE_AA)

        for i, k in enumerate(info.keys()):
            v = info[k]
            key_text = '{}: '.format(k)
            (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_size, thickness)
            cv2.putText(frame, key_text,
                        (x, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (66, 133, 244), thickness, cv2.LINE_AA)
            cv2.putText(frame, str(v),
                        (x + key_width, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def runmp(env, planner, goal_site='target'):
    ee_error = 0

    # Create env to simulate true motion plan, and one for IK
    mp_env = gym.make(args.env, **args.__dict__)
    mp_env.reset()
    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()


    qpos = env.sim.data.qpos.ravel().copy()
    qvel = env.sim.data.qvel.ravel().copy()
    success = False
    mp_env.set_state(qpos, qvel)
    ik_env.set_state(qpos, qvel)


    # Obtain start and goal joint positions. Do IK to get joint positions for goal_site.
    result = qpos_from_site_pose_sampling(ik_env, 'grip_site', target_pos=env._get_pos(goal_site),
                                            target_quat=env._get_quat(goal_site), joint_names=env.model.robot.joints, max_steps=1000, tol=1e-3)
    logger.info("IK for %s successful? %s. Err_norm %.5f", goal_site, result.success, result.err_norm)

    # Equate qpos components not affected by planner
    start = qpos
    goal = start.copy()
    goal[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes]
    ik_env.set_state(goal, qvel)

    # visual inspection (TODO: Fix. doesn't work with saving video the second time)
    # ik_env.render('human')
    # input("See if IK solution is fine. Press any key to continue; Ctrl-C to quit")

    # OMPL Planning
    traj, _ = planner.plan(start, goal,  args.timelimit, args.max_meta_len)

    # Success condition
    if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
        success = True
        logger.info("Planner to %s success", goal_site)
    else:
        logger.error("Planner to %s failed", goal_site)
        exit(1)


    frames = []
    step = 0

    if success:
        num_joints = len(env.model.robot.joints)
        for step, state in enumerate(traj[1:]):
            qpos = env.sim.data.qpos
            qvel = env.sim.data.qvel
            new_state = np.concatenate((traj[step + 1][:num_joints], qpos[num_joints:])).ravel().copy()
            mp_env.set_state(new_state, qvel.ravel())
            if is_save_video:
                frames.append(render_frame(env, step))
            else:
                env.render(mode='human')

            # Change indicator robot position
            # env.set_robot_indicator_joint_positions(state[env.ref_joint_pos_indexes])

            env.sim.data.qpos[:] = state
            env.sim.forward()

            ee_error += np.sqrt((env.data.get_site_xpos('grip_site') - mp_env.data.get_site_xpos('grip_site')) ** 2)

    if is_save_video:
        frames.append(render_frame(env, step))
    else:
        env.render(mode='human')


    num_states = len(traj[1:])
    return frames, num_states, ee_error/num_states, success

# MAIN SCRIPT

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
else:
    raise ValueError('args.env (%s) is not supported for this test script' % args.env)

add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

# Save video or not
is_save_video = True
record_caption = True

env = gym.make(args.env, **args.__dict__)
non_limited_idx = np.where(env._is_jnt_limited==0)[0]
## Create planner

# Set contacts to ignore
boxid = env.sim.model.geom_name2id('box')
tableid = env.sim.model.geom_name2id('table_collision')
ignored_contacts = [make_ordered_pair(boxid, tableid)]
logger.info("Ignored contacts\t%s", ignored_contacts)

# Set joints to ignore
passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]
logger.info("Passive joint IDs\t%s", passive_joint_idx)

# Check if initial state is valid to pass to the planner
env.reset()
if 'pick-move' in args.env:
    env.sim.data.qpos[13:17] = [0., 1., 0., 0.] # TODO: Make less hacky (maybe when creating the robosuite env?)
    # env._set_quat('box', [0., 1., 0., 0.])
env.sim.forward() # set positions
valid = check_state_validity(env, ignored_contacts)

if valid:
    logger.info("Valid start state for motion plan. Initializing planner")
    planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx, passive_joint_idx=passive_joint_idx, ignored_contacts=ignored_contacts)
else:
    logger.error("Invalid start state for the planner. Quitting")
    exit(1)

## Now run planner to reach pre-grasping position
frames1, num_states, _, success = runmp(env, planner, goal_site='box')
logger.info('Pregrasp success? %s', success)

## Now close the gripper
ac = np.zeros(env.dof)
ac[-1] = -1.
for step in range(2):
    env.step(ac)
    frames1.append(render_frame(env, step))

# visual inspection
# env.render('human')
# input("See if gripper is closed. Press any key to continue; Ctrl-C to quit")

## Post-grasp checks
glue_bodies = [b"right_gripper_base", b"box"]
logger.info("Glue bodies\t%s", glue_bodies)

# Check if grasped state is valid
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('r_finger_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('r_fingertip_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('l_finger_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('l_fingertip_g0')))
logger.info("New ignored contacts after grasp\t%s", ignored_contacts)
valid = check_state_validity(env, ignored_contacts)

if valid:
    logger.info("Valid start state for motion plan. Initializing planner")
    planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx, passive_joint_idx=passive_joint_idx, glue_bodies=glue_bodies, ignored_contacts=ignored_contacts)
else:
    logger.error("Invalid start state for the planner. Quitting")
    exit(1)

## Now move to target
frames2, num_states, _, success = runmp(env, planner, goal_site='target')
logger.info('Postgrasp success: %s' % (success))

# Create video
if is_save_video:
    frames = frames1 + frames2
    prefix_path = os.path.join('./tmp', args.planner_type, args.env, str(args.construct_time))
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    fpath = os.path.join(prefix_path, '{}-{}-{}-timelimit_{}-threshold_{}-range_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range))
    save_video(fpath, frames, fps=5)
