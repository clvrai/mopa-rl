import env
import gym
from config import argparser
import numpy as np
from motion_planners.sampling_based_planner import SamplingBasedPlanner
from util.gym import action_size, observation_size
from config.motion_planner import add_arguments as planner_add_arguments
import time

parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'mover' in args.env:
    from config.mover import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser) # add env arguments
planner_add_arguments(parser) # add planner arguments
args, unparsed = parser.parse_known_args() # add these new arguments to `args`

env = gym.make(args.env, **args.__dict__)
if 'robosuite' in args.env:
    env.use_camera_obs = False
obs = env.reset()
print(args)

print('robot_joints:  ', env.ref_joint_pos_indexes)
print('l gripper_ids: ', env.l_finger_geom_ids)
print('r gripper_ids: ', env.r_finger_geom_ids)
print('num actions:   ', action_size(env.action_space))
print('num state:     ', observation_size(env.observation_space))
print('dof:           ', env.dof)
print('len(qpos):     ', len(env.sim.data.qpos))

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]
print(passive_joint_idx)

glue_bodies = [b"gripper_base", b"box"]

planner = SamplingBasedPlanner(config=args,
                               xml_path=env.xml_path,
                               num_actions=action_size(env.action_space),
                               passive_joint_idx=passive_joint_idx,
                               glue_bodies=glue_bodies)

start = env.sim.data.qpos.ravel().copy()*0.0
start[7] = 0.35
goal = start.copy()
goal[0] = 2.5
goal[1] = 2.5
goal[2] = 2.5
print('start ', start)
print('goal ', goal)
states = planner.planner.plan(start, goal, args.timelimit, 40)
traj = np.array(states)

print('traj', traj)
print(traj.shape)

# replay path
while True:
    for t in range(traj.shape[0]):
        env.sim.data.qpos[:] = traj[t, :]
        env.sim.forward()
        env.render(mode='human')
        time.sleep(0.1)
