import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video, render_frame, mujocopy_render_hack
from config.motion_planner import add_arguments as planner_add_arguments
from env.inverse_kinematics import qpos_from_site_pose_sampling
from util.gym import render_frame
import cv2
import time
import timeit
import copy
np.set_printoptions(precision=3)

mujocopy_render_hack() # workaround for mujoco py issue #390

def interpolate(env, next_qpos, out_of_bounds, planner):
    interpolated_traj = []
    current_qpos = env.sim.data.qpos

    min_action = env.action_space.spaces['default'].low[0] * env._ac_scale * 0.8  # assume equal for all
    max_action = env.action_space.spaces['default'].high[0] * env._ac_scale * 0.8 # assume equal for all
    assert max_action > min_action, "action space box is ill defined"
    assert max_action > 0 and min_action < 0, "action space MAY be ill defined. Check this assertion"

    action = env.form_action(next_qpos)
    action_arr = action['default']

    # Step1: get scaling factor. get scaled down action within action limits
    scaling_factor = 1
    for i in out_of_bounds:
        ac = action_arr[i]
        sf = ac/max_action if (ac > max_action) else ac/min_action # assumes max>0, min<0 !! Check signs!
        scaling_factor = max(scaling_factor, sf)

    scaled_ac = action_arr[:len(env.ref_joint_pos_indexes)]/scaling_factor
    action['default'] = scaled_ac

    # Step2: Run scaled down action for floor(scaling factor) steps
    interp_qpos = current_qpos.copy()
    valid = True
    for i in range(int(scaling_factor)): # scaling_factor>0 => int(scaling_factor) == int(floor(scaling_factor))
        interp_qpos[env.ref_joint_pos_indexes] += scaled_ac
        if not planner.isValidState(interp_qpos):
            valid = False
            break
        interpolated_traj.append(interp_qpos.copy())
    if not valid:
        interpolated_traj = planner.plan(env.sim.data.qpos, next_qpos, args.simple_planner_timelimit)[0]
    else:
        interpolated_traj.append(next_qpos)
    print("Curr qpos %s,\torig. action %s,\tscaled down action %s" %(current_qpos[env.ref_joint_pos_indexes], action_arr, scaled_ac))
    print("Set of interpolated states\n\t", [qpos[env.ref_joint_pos_indexes] for qpos in interpolated_traj])

    return interpolated_traj

parser = argparser()
args, unparsed = parser.parse_known_args()
if 'pusher' in args.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
    add_arguments(parser)
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
elif 'reacher' in args.env:
    from config.reacher import add_arguments
    add_arguments(parser)

planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)
args._xml_path = env.xml_path

ignored_contacts = []
# Allow collision with manipulatable object
geom_ids = env.agent_geom_ids
for manipulation_geom_id in env.manipulation_geom_ids:
    for geom_id in geom_ids:
        ignored_contacts.append(make_ordered_pair(manipulation_geom_id, geom_id))

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]

non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts, is_simplified=args.is_simplified, simplified_duration=args.simplified_duration) # default goal bias is 0.05
simple_planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts, goal_bias=1.0, is_simplified=False)


N = 1
is_save_video = True
frames = []
# TODO: This code is repeated in interpolate(). Fix this
min_action = env.action_space.spaces['default'].low[0] * env._ac_scale * 0.8  # assume equal for all
max_action = env.action_space.spaces['default'].high[0] * env._ac_scale * 0.8 # assume equal for all
assert max_action > min_action, "action space box is ill defined"
assert max_action > 0 and min_action < 0, "action space MAY be ill defined. Check this assertion"

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
        goal_joint_pos = get_goal_position(env, goal_site='cube')
        target_qpos = current_qpos.copy()
        target_qpos[env.ref_joint_pos_indexes] += np.random.uniform(low=-1, high=1, size=len(env.ref_joint_pos_indexes))
        if not simple_planner.isValidState(target_qpos): # check whether a target is valid or not
            env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
            if is_save_video:
                frames[episode].append(render_frame(env, step))
            else:
                env.render("human")
            print("Invalid goal state")
            step += 1
            continue
        else:
            print("Valid goal state")

        traj, success, valid, exact = simple_planner.plan(current_qpos, target_qpos, timelimit=args.simple_planner_timelimit)
        env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
        xpos = OrderedDict()
        xquat = OrderedDict()

        if not success and not exact:
            traj, success, valid, exact = planner.plan(current_qpos, target_qpos)
            print("Using normal planner path (%d points)" % len(traj))
        else:
            print("Using simpler planner path (%d points)" % len(traj))
        print("==============")

        if is_save_video:
            frames[episode].append(render_frame(env, step))
        else:
            env.render('human')

        reward = 0
        if success:
            for j, next_qpos in enumerate(traj):

                # move a next_qpos if it is invalid state
                trial = 0
                while not planner.isValidState(next_qpos) and trial < 100:
                    d = env.sim.data.qpos.copy()-next_qpos
                    next_qpos += 0.05 * d/np.linalg.norm(d)
                    trial+=1

                action = env.form_action(next_qpos)
                action_arr = action['default']
                out_of_bounds = np.where((action_arr[:len(env.ref_joint_pos_indexes)] > max_action) | (action_arr[:len(env.ref_joint_pos_indexes)]<min_action))[0]

                env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())
                if len(out_of_bounds) > 0: #Some actions out of bounds
                    reward = 0
                    # interpolate
                    interpolated_traj = interpolate(env, next_qpos, out_of_bounds, simple_planner)
                    for interp_qpos in interpolated_traj:
                        if not planner.isValidState(interp_qpos):
                            print("Interpolated state %s is invalid!! Still stepping\n" % interp_qpos[env.ref_joint_pos_indexes])
                        action = env.form_action(interp_qpos)
                        step += 1
                        ob, interp_reward, done, info = env.step(action, is_planner=True)
                        if is_save_video:
                            info['ac'] = action['default']
                            info['next_qpos'] = next_qpos
                            info['target_qpos'] = target_qpos
                            info['curr_qpos'] = env.sim.data.qpos.copy()
                            frames[episode].append(render_frame(env, step, info))
                        else:
                            env.render('human')
                        reward += interp_reward
                        if done:
                            break
                    action = env.form_action(next_qpos)
                    env._reset_prev_state()
                else:
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
                if done:
                    break
        else:
            step += 1
            if step > args.max_episode_steps:
                break
            if is_save_video:
                frames[episode].append(render_frame(env, step))
            else:
                env.render('human')


if is_save_video:
    prefix_path = './tmp/motion_planning_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    for i, episode_frames in enumerate(frames):
        fpath = os.path.join(prefix_path, 'test_trial_{}.mp4'.format(i))
        save_video(fpath, episode_frames, fps=5)
