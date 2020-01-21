import numpy as np

from motion_planners.planner import PyPlanner


planner = PyPlanner('../mujoco-ompl/problems/reacher.xml'.encode('utf-8'))
start = [0, 0, 0.1, 0.1, 0, 0, 0, 0]
goal =  [1.5, 0.5, 0.1, 0.1, 0, 0, 0, 0]
print(np.array(planner.planning(start, goal, 1.)))
