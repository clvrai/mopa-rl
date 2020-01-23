import os, sys

import numpy as np
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyPlanner


class SamplingBasedPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions, config.sst_selection_radius, config.sst_pruning_radius, config.planner_objective.encode('utf-8'))

    def plan(self, start, goal, planner='sst', timelimit=1.):
        traj = np.array(self.planner.planning(start, goal, timelimit))
        return traj

    def plan_control(self, start, goal, planner='sst', timelimit=1.):
        actions = np.array(self.planner.planning_control(start, goal, timelimit))
        return actions

    def kinematic_plan(self, start, goal, timelimit=1., _range=0.1):
        control = np.array(self.planner.kinematic_planning(start, goal, timelimit, _range))
        return control


