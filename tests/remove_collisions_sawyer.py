import os, sys
import numpy as np
import shutil

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
                print("Contact pair (%d, %d) not in list of contacts to ignore. Invalid state." % (con_pair[0], con_pair[1]))
                valid = False
                break
    return valid

def render_frame(env, step, info={}):
    color = (200, 200, 200)
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


def runmp(env, planner, goal_site='target', i=None):  
    end_error = 0

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

    print("IK for %s successful? %s. Err_norm %.5f" % (goal_site, result.success, result.err_norm))
    # Equate qpos components not affected by planner
    start = qpos
    goal = start.copy()
    goal[env.ref_joint_pos_indexes] = result.qpos[env.ref_joint_pos_indexes]
    ik_env.set_state(goal, qvel)
    # print(goal[env.ref_joint_pos_indexes])
    ik_env.render('human')
    input("See if IK solution is fine. Press any key to continue; Ctrl-C to quit")

    # OMPL Planning
    traj, _ = planner.plan(start, goal,  args.timelimit, args.max_meta_len)

    # Success condition
    if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
        success = True
        print("Planner success")
    else:
        print("Planner failure")


    frames = []
    step = 0

    if success:
        # goal = env.sim.data.qpos[-2:]
        # prev_state = traj[0, :]
        # i_term = np.zeros_like(env.sim.data.qpos[:-2])
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
            action = state - qpos.copy()
            action = np.concatenate([action[env.ref_joint_pos_indexes], np.array([0])])

            # env.step(action)
            pos = start.copy()
            pos[env.ref_joint_pos_indexes] = state[env.ref_joint_pos_indexes]
            env.set_state(pos, env.sim.data.qvel)

            end_error += np.sqrt((env.data.get_site_xpos('grip_site') - mp_env.data.get_site_xpos('grip_site')) ** 2)
            prev_state = state

    if is_save_video:
        frames.append(render_frame(env, step))
        prefix_path = os.path.join('./tmp', args.planner_type, args.env, str(args.construct_time))
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        if i is None:
            i == ""
        fpath = os.path.join(prefix_path, '{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
        save_video(fpath, frames, fps=5)
    else:
        env.render(mode='human')


    num_states = len(traj[1:])
    if num_states == 0:
        return 0, num_states, 0, success
    else:
        return -1, num_states, end_error/len(traj[1:]), success

# MAIN SCRIPT

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'mover' in args.env:
    from config.mover import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
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
# Create planner
boxid = env.sim.model.geom_name2id('box')
tableid = env.sim.model.geom_name2id('table_collision')
ignored_contacts = [make_ordered_pair(boxid, tableid)]
print("Ignored contacts\t", ignored_contacts)

# Check if initial state is valid to pass to the planner
env.reset()
if 'pick-move' in args.env:
    env.sim.data.qpos[13:17] = [0., 1., 0., 0.] # TODO: Make less hacky
    # env._set_quat('box', [0., 1., 0., 0.])
env.sim.forward() # set positions
valid = check_state_validity(env, ignored_contacts)

if valid:
    print("Valid start state for motion plan. Initializing planner")
    planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx, ignored_contacts=ignored_contacts)
else:
    print("Invalid start state for the planner. Quitting")
    exit(1)

## Now run planner to reach pre-grasping position
_, num_states, end_error, success = runmp(env, planner, goal_site='box')
print('Planner success: %s' % (success))

## Now close the gripper
ac = np.zeros(env.dof)
ac[-1] = -1.
for i in range(2):
    env.step(ac)
# visual inspection before continuing
env.render('human')
input("See if gripper is closed. Press any key to continue; Ctrl-C to quit")

## Post-grasp checks
# Check if grasped state is valid
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('r_finger_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('r_fingertip_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('l_finger_g0')))
ignored_contacts.append(make_ordered_pair(boxid, env.sim.model.geom_name2id('l_fingertip_g0')))
valid = check_state_validity(env, ignored_contacts)

if valid:
    print("Valid start state for motion plan. Initializing planner")
    planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx, ignored_contacts=ignored_contacts)
else:
    print("Invalid start state for the planner. Quitting")
    exit(1)

## Now move to target
_, num_states, end_error, success = runmp(env, planner, goal_site='target')

# global_num_states = 0
# global_end_error = 0
# N = 1
# num_success = 0
# for i in range(N):
#     _, num_states, end_error, success = runmp(env, planner)
#     global_end_error += end_error
#     global_num_states += num_states
#     if success:
#         num_success += 1

# print('Planner success: %d out of %d times' % (num_success, N))
# print('End effector error: ', global_end_error/N)
