import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video
from config.motion_planner import add_arguments as planner_add_arguments
import cv2
import time
import timeit


def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

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





parser = argparser()
args, unparsed = parser.parse_known_args()
if 'pusher' in args.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
    add_arguments(parser)

planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

env = gym.make(args.env, **args.__dict__)
args._xml_path = env.xml_path
args.planner_type="sst"
args.planner_objective="path_length"
args.range = 0.05
args.threshold = 0.05
args.timelimit = 1.0
args.contact_threshold = -0.001

ignored_contacts = []
# Allow collision with manipulatable object
geom_ids = env.agent_geom_ids
for manipulation_geom_id in env.manipulation_geom_ids:
    for geom_id in geom_ids:
        ignored_contacts.append(make_ordered_pair(manipulation_geom_id, geom_id))

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]

non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts) # default goal bias is 0.05
# planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts, is_simplified=True, simplified_duration=0.5) # default goal bias is 0.05
simple_planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts, 1.0)


N = 1
is_save_video = False
frames = []
# start_pos = np.array([-2.77561, 0.106835, 0.047638, -0.15049436,  0.16670527, -0.00635442, 0.14496655])
# ob = env.reset()
# env.set_state(start_pos, env.sim.data.qvel.copy())

for episode in range(N):
    print("Episode: {}".format(episode))
    done = False
    ob = env.reset()
    # env.set_state(start_pos, env.sim.data.qvel.copy())
    step = 0
    if is_save_video:
        frames.append([render_frame(env, step)])
    else:
        env.render('human')

    while not done:
        current_qpos = env.sim.data.qpos.copy()
        target_qpos = current_qpos.copy()
        # target_qpos[env.ref_joint_pos_indexes] = np.array([-0.35, -0.986, -0.667])
        target_qpos[env.ref_joint_pos_indexes] += np.random.uniform(low=-2., high=2., size=len(env.ref_joint_pos_indexes))
        # target_qpos[env.ref_joint_pos_indexes] = np.ones(len(env.ref_joint_pos_indexes)) * 0.5 # you can reproduce the invalid goal state
        traj, success, valid, exact = simple_planner.plan(current_qpos, target_qpos, timelimit=0.01)
        env.visualize_goal_indicator(target_qpos[env.ref_joint_pos_indexes].copy())
        xpos = OrderedDict()
        xquat = OrderedDict()

        if not success:
            traj, success, valid, exact = planner.plan(current_qpos, target_qpos)
            print("Normal planner is called")

            if not success and not exact:
                print("Approximate")
            elif not success and not valid:
                print("Invalid state")
            else:
                print("Success")
        else:
            print("Interpolation")
        # traj, success = planner.plan(current_qpos, target_qpos)
        print("==============")
        print(step)

        if success:
            for j, next_qpos in enumerate(traj):
                action = env.form_action(next_qpos)
                ob, reward, done, info = env.step(action, is_planner=True)

                env.visualize_dummy_indicator(next_qpos[env.ref_joint_pos_indexes].copy())

                step += 1
                if is_save_video:
                    frames[episode].append(render_frame(env, step))
                else:
                    import timeit
                    t = timeit.default_timer()
                    while timeit.default_timer() - t < 0.1:
                        env.render('human')
                if done:
                    break
        # else:
        #     env.render('human')


if is_save_video:
    prefix_path = './tmp/motion_planning_test/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    for i, episode_frames in enumerate(frames):
        fpath = os.path.join(prefix_path, 'test_trial_{}.mp4'.format(i))
        save_video(fpath, episode_frames, fps=5)
