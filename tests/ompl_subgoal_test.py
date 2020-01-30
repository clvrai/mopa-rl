import os, sys
import numpy as np
import shutil

from motion_planners.sampling_based_planner import SamplingBasedPlanner
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

is_save_video = True
record_caption = True

env = gym.make(args.env, **args.__dict__)
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
    # mp_env = gym.make(args.env, **args.__dict__)
    # mp_env.reset()
    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()
    qpos = env.sim.data.qpos.ravel()
    qvel = env.sim.data.qvel.ravel()
    env.goal = [-0.25128643, 0.14829235]
    qpos[-2:] = env.goal
    qpos[:-2] = [0.09539838, 0.04237122, 0.05476331, -0.0676346, -0.0434791, -0.06203809, 0.03571644]
    qvel[:-2] = [ 0.00293847, 0.00158573, 0.0018593, 0.00122192, -0.0016253, 0.00225007, 0.00001702]


    env._set_pos('subgoal1', [0.2, 0.05,0])
    env._set_pos('subgoal2', [0.03, 0.1,0])
    env._set_pos('subgoal3', [-0.18, 0.05,0])

    env.set_state(qpos, qvel)

    success_count = 0
    success = False
    goal = env.goal

    frames = []
    num_states = 0
    subgoals = ['subgoal1', 'subgoal2', 'subgoal3', 'target']
    for subgoal in subgoals:
        ik_env.set_state(env.sim.data.qpos.ravel(), env.sim.data.qvel.ravel())
        result = qpos_from_site_pose_sampling(ik_env, 'fingertip', target_pos=env._get_pos(subgoal), target_quat=env._get_quat(subgoal), joint_names=env.model.joint_names[:-2], max_steps=100)

        goal = result.qpos

        start = env.sim.data.qpos.ravel()
        #planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space))
        traj, actions = planner.plan(start, goal,  args.timelimit, max_steps=50)
        print(start ,goal)
        if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
            success = True

        if success:
            for step, state in enumerate(traj[1:]):
                if is_save_video:
                    frames.append(render_frame(env, num_states+step))
                else:
                    env.render(mode='human')
                env.set_state(np.concatenate((state[:-2], env.sim.data.qpos.ravel()[-2:])).ravel(), env.sim.data.qvel.ravel())
                #mp_env.set_state(np.concatenate((state[:-2], env.sim.data.qpos.ravel()[-2:])).ravel(), env.sim.data.qvel.ravel())
                #env.step(-(env.sim.data.qpos[:-2]-state[:-2])*env._frame_skip)
                action = state[:-2]-env.get_joint_positions
                #env.step(action)
                #error += np.sqrt((env.sim.data.qpos - state)**2)
                #end_error += np.sqrt((env.data.get_site_xpos('fingertip')-mp_env.data.get_site_xpos('fingertip'))**2)
            num_states += len(traj[1:])
            success_count += 1
        else:
            env.set_state(np.concatenate((result.qpos[:-2], env.goal)), env.sim.data.qvel.ravel())


    if is_save_video:
        frames.append(render_frame(env, num_states))
        prefix_path = os.path.join('./tmp', args.planner_type, args.env, str(args.construct_time))
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        if i is None:
            i == ""
        fpath = os.path.join(prefix_path, '{}-{}-{}-timelimit_{}-threshold_{}-range_{}_{}.mp4'.format(args.env, args.planner_type, args.planner_objective, args.timelimit, args.threshold, args.range, i))
        save_video(fpath, frames, fps=5)
    else:
        env.render(mode='human')



    return 0, 0, 0, success_count
    #return success

N = 10
num_success = 0
for i in range(N):
    error, num_states, end_error, success = run_mp(env, planner, i)
    num_success += success

print("Success: ", num_success)
print('End effector error: ', end_error)
print('Joint state error: ', error)




