import os, sys

import numpy as np
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyKinematicPlanner
from motion_planners.kino_planner import PyKinodynamicPlanner
from util.env import joint_convert


class SamplingBasedPlanner:
    def __init__(self, config, xml_path, num_actions, non_limited_idx=None):
        self.config = config
        self.planner = PyKinematicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius,
                                 config.sst_selection_radius,
                                 config.planner_objective.encode('utf-8'),
                                 config.threshold,
                                 config.range,
                                 config.construct_time)
        self.non_limited_idx = non_limited_idx

    def convert_nonlimited(self, state):
        for idx in self.non_limited_idx:
            state[idx] = joint_convert(state[idx])
        return state

    def plan(self, start, goal, timelimit=1., max_steps=200):
        converted_start = self.convert_nonlimited(start.copy())
        converted_goal = self.convert_nonlimited(goal.copy())
        states = np.array(self.planner.plan(converted_start, converted_goal, timelimit, max_steps))

        traj = [start]
        pre_state = states[0]
        for i, state in enumerate(states[1:]):
            #converted_pre_state = self.convert_nonlimited(pre_state.copy())
            tmp_state = traj[-1] + (state - pre_state)
            for idx in self.non_limited_idx:
                if abs(state[idx]-pre_state[idx]) > 3.14:
                    if pre_state[idx] > 0 and state[idx] <= 0:
                        tmp_state[idx] = traj[-1][idx] + (3.14-pre_state[idx] + state[idx] + 3.14)
                    elif pre_state[idx] < 0 and state[idx] > 0:
                        tmp_state[idx] = traj[-1][idx] + (3.14-state[idx] + pre_state[idx] + 3.14)
            pre_state = tmp_state
            traj.append(tmp_state)
        return np.array(traj), states


class SamplingBasedKinodynamicPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyKinodynamicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius, config.sst_pruning_radius)

    def plan(self, start, goal, timelimit=1.):
        controls = np.array(self.planner.plan(start, goal, timelimit))

        # TODO more efficient way
        return controls


