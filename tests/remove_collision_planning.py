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



def run_mp(env, planner, i=None):
    error = 0
    end_error = 0
    env.reset()
    init_qpos = np.array([1.14, -1.91, -1, 0., 0., -0.068, -0.1, 0.0851, -0.125])
    env.set_state(init_qpos, np.zeros_like(env.sim.data.qvel.ravel().copy()))
    env.render('human')
    ac = np.zeros(4)
    ac[-1] = 1.
    for _ in range(2):
        env.step(ac)

    mp_env = gym.make(args.env, **args.__dict__)
    mp_env.reset()
    qpos = env.sim.data.qpos.ravel()
    qvel = env.sim.data.qvel.ravel()
    success = False
    env.set_state(qpos, qvel)
    goal = env.goal


    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()
    ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    env_prime.reset()
    env_prime.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())

    # IK to find a goal state
    result = qpos_from_site_pose_sampling(ik_env, 'grip_site', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.sim.model.joint_names[:3], max_steps=300)
    ik_env.set_state(result.qpos, ik_env.sim.data.qvel.ravel())

    # Update dummy reacher states (goal state and ompl states)

    start = env.sim.data.qpos.ravel().copy()
    start[2:4] = np.clip(start[2:4], env.sim.model.jnt_range[2:4][0], env.sim.model.jnt_range[2:4][1])
    goal = start.copy()
    goal[:3] = result.qpos[:3]


    # OMPL Planning
    traj, _ = planner.plan(start, goal,  args.timelimit, args.max_meta_len)

    # Success condition
    if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
        success = True
        print("Success")
    else:
        print("Failure")




    frames = []
    action_frames = []
    step = 0

    sim_dt = 0.01
    edge_dt = 1.
    Kp = 150.0
    Kd = 20.0
    Ki = 0.1
    alpha = 0.95
    n_inner_loop = int(edge_dt / sim_dt)

    if success:
        goal = env.sim.data.qpos[-2:]
        prev_state = traj[0, :]
        i_term = np.zeros_like(env.sim.data.qpos[:-2])

        for step, state in enumerate(traj[1:]):

            # Update dummy reacher
            if step % 1 == 0:
                mp_env.set_state(np.concatenate((traj[step + 1][:-2], goal)).ravel(), env.sim.data.qvel.ravel())

            if is_save_video:
                frames.append(render_frame(env, step))
            else:
                env.render(mode='human')

            ac = np.ones(4)
            ac[:3] = state[:3] - env.sim.data.qpos[:3].copy()
            env.step(ac)

            error += np.sqrt((env.sim.data.qpos - state) ** 2)
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
        return error / len(traj[1:]), num_states, end_error/len(traj[1:]), success


parser = argparser()
args, unparsed = parser.parse_known_args()

if 'mover' in args.env:
    from config.mover import add_arguments
else:
    raise ValueError('args.env (%s) is not supported for this test script' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

# Save video or not
is_save_video = True
record_caption = True

env = gym.make(args.env, **args.__dict__)
env_prime = gym.make(args.env, **args.__dict__)
non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx)
box_geom_id = env.sim.model.geom_name2id('box')
planner.remove_collision(box_geom_id, 0, 0)

errors = 0
global_num_states = 0
global_end_error = 0
N = 20
num_success = 0
error, num_states, end_error, success = run_mp(env, planner)
errors += error
global_end_error += end_error
global_num_states += num_states
if success:
    num_success += 1

print(num_success)
print('End effector error: ', global_end_error/N)
print('Joint state error: ', errors/N)
