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
from util.gym import action_size
import time
import cv2
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

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
    mp_env = gym.make(args.env, **args.__dict__)
    mp_env.reset()
    ik_env = gym.make(args.env, **args.__dict__)
    ik_env.reset()

    qpos = env.sim.data.qpos.ravel().copy()
    qvel = env.sim.data.qvel.ravel().copy()
    success = False
    mp_env.set_state(qpos, qvel)
    ik_env.set_state(qpos, qvel)

    goal_site = 'target'
    result = qpos_from_site_pose_sampling(ik_env, 'grip_site', target_pos=env._get_pos(goal_site),
    target_quat=env._get_quat(goal_site), joint_names=env.model.robot.joints, max_steps=1000, tol=1e-2)

    print("IK for %s successful? %s. Err_norm %.3f" % (goal_site, result.success, result.err_norm))

    start = env.sim.data.qpos.ravel().copy()
    goal = result.qpos
    goal[len(env.model.robot.joints):] = start[len(env.model.robot.joints):]
    ik_env.set_state(np.concatenate((result.qpos[:len(env.model.robot.joints)], env.sim.data.qpos[len(env.model.robot.joints):])).ravel().copy(), env.sim.data.qvel.ravel())
    # OMPL Planning
    traj, _ = planner.plan(start, goal,  args.timelimit, 40)

    # Success condition
    if len(np.unique(traj)) != 1 and traj.shape[0] != 1:
        success = True
        print("Planner succeeded in planning trajectory to %s!" % goal_site)

    frames = []
    action_frames = []
    step = 0

    if success:
        goal = env.sim.data.qpos[-2:]
        prev_state = traj[0, :]
        i_term = np.zeros_like(env.sim.data.qpos[:-2])

        for step, state in enumerate(traj[1:]):

            if is_save_video:
                frames.append(render_frame(env, step))
            else:
                env.render(mode='human')

            # Change indicator robot position
            env.set_robot_indicator_joint_positions(state[:len(env.model.robot.joints)])

            action = state-env.sim.data.qpos.copy()
            action = action[:len(env.model.robot.joints)+1]
            action[-1] = 0.
            env.step(action)

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
record_caption = True

env = gym.make(args.env, **args.__dict__)

env.use_camera_obs = False
non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = SamplingBasedPlanner(args, env.xml_path, action_size(env.action_space), non_limited_idx)

errors = 0
global_num_states = 0
global_end_error = 0
N = 1
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
