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

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)


add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

is_save_video = False
record_caption = True

env = gym.make(args.env, **args.__dict__)
env_prime = gym.make(args.env, **args.__dict__)
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space))

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
    print("After reset: ", env.sim.data.ncon)
    mp_env = gym.make(args.env, **args.__dict__)
    mp_env.reset()
    qpos = env.sim.data.qpos.ravel()
    qvel = env.sim.data.qvel.ravel()
    #env.goal = [-0.25128643, 0.14829235]
    #qpos[-2:] = env.goal
    #qpos[:-2] = [0.09539838, 0.04237122, 0.05476331, -0.0676346, -0.0434791, -0.06203809, 0.03571644]
    #qvel[:-2] = [ 0.00293847, 0.00158573, 0.0018593, 0.00122192, -0.0016253, 0.00225007, 0.00001702]
    success = False
    mp_env.set_state(qpos, qvel)
    goal = env.goal


    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()
    ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    env_prime.reset()
    env_prime.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
    #result = qpos_from_site_pose(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=1000)
    result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=env._get_pos('target'), target_quat=env._get_quat('target'), joint_names=env.model.joint_names[:-2], max_steps=500)
    ik_env.set_state(result.qpos, ik_env.sim.data.qvel.ravel())
    for l in range(3):
        body_idx = ik_env.model.body_name2id('body'+str(l))
        pos = ik_env.sim.data.body_xpos[body_idx]
        quat = ik_env.sim.data.body_xquat[body_idx]
        env._set_pos('body'+str(l)+'-goal', pos)
        env._set_quat('body'+str(l)+'-goal', quat)

    for l in range(3):
        body_idx = env.model.body_name2id('body'+str(l))
        pos = env.sim.data.body_xpos[body_idx]
        quat = env.sim.data.body_xquat[body_idx]
        env._set_pos('body'+str(l)+'-dummy', pos)
        env._set_quat('body'+str(l)+'-dummy', quat)


    start = env.sim.data.qpos.ravel()
    goal = result.qpos
    print(start)

    traj, _ = planner.plan(start, goal,  args.timelimit, args.max_meta_len)
    if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
        success = True

    frames = []
    action_frames = []
    step = 0
    print('After plan: ', env.sim.data.ncon)
    info = {}
    if success:
        goal = env.sim.data.qpos[-2:]
        for step, state in enumerate(traj[1:]):
            if is_save_video:
                 frames.append(render_frame(env, step, info))
            else:
                env.render(mode='human')

            if step % 1 == 0:
                mp_env.set_state(np.concatenate((state[:-2], goal)).ravel(), env.sim.data.qvel.ravel())
                for l in range(3):
                    body_idx = mp_env.model.body_name2id('body'+str(l))
                    pos = mp_env.sim.data.body_xpos[body_idx]
                    quat = mp_env.sim.data.body_xquat[body_idx]
                    env._set_pos('body'+str(l)+'-dummy', pos)
                    env._set_quat('body'+str(l)+'-dummy', quat)

            action = state[:-2] - env.sim.data.qpos[:-2]
            #env.set_state(np.concatenate((state[:-2], goal)).ravel(), env.sim.data.qvel.ravel())
            env.step(action)
            info['state'] = state[:-2]
            info['curr_pos'] =  env.sim.data.qpos.ravel()[:-2]



            error += np.sqrt((env.sim.data.qpos - state)**2)
            end_error += np.sqrt((env.data.get_site_xpos('fingertip')-mp_env.data.get_site_xpos('fingertip'))**2)
    else:
        if is_save_video:
             frames.append(render_frame(env, step, info))
        else:
            env.render(mode='human')


    if is_save_video:
        frames.append(render_frame(env, step, info))
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
    print("num states")

    if num_states == 0:
        return 0, num_states, 0, success
    else:
        return error / len(traj[1:]), num_states, end_error/len(traj[1:]), success


    #return success

errors = 0
global_num_states = 0
global_end_error = 0
N = 20
num_success = 0
for i in range(N):
    error, num_states, end_error, success = run_mp(env, planner, i)
    errors += error
    global_end_error += end_error
    global_num_states += num_states
    if success:
        num_success += 1

print(num_success)
print('End effector error: ', global_end_error/N)
print('Joint state error: ', errors/N)


