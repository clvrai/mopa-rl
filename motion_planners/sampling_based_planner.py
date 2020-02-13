import os, sys

import numpy as np
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyKinematicPlanner
from motion_planners.kino_planner import PyKinodynamicPlanner


class SamplingBasedPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyKinematicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius,
                                 config.sst_selection_radius,
                                 config.planner_objective.encode('utf-8'),
                                 config.threshold,
                                 config.range,
                                 config.construct_time)

    def plan(self, start, goal, timelimit=1., max_steps=200):
        states = np.array(self.planner.plan(start, goal, timelimit, max_steps))

        traj = [states[0]]
        pre_state = states[0]
        for i, state in enumerate(states[1:]):
            tmp_state = pre_state + (state-pre_state)
            if abs(state[0]-pre_state[0]) > 3.14:
                if pre_state[0] > 0 and state[0] < 0:
                    tmp_state[0] = 3.14 + (3.14+state[0])
                elif pre_state[0] < 0 and state[0] > 0:
                    tmp_state[0] = -3.14 + (-3.14+state[0])
            # if state[0]-pre_state[0] < -3.14:
            #     tmp_state[0] = pre_state[0] + (3.14-states[i][0] + state[0]+3.14)
            #     #tmp_state[1:] = pre_state[1:] + (state[1:]-pre_state[1:])
            # elif state[0]-pre_state[0]> 3.14:
            #     tmp_state[0] = pre_state[0] - (3.14-state[0] + 3.14+states[i][0])
            #     #tmp_state[1:] = pre_state[1:] + (state[1:]-pre_state[1:])
            pre_state = tmp_state
            traj.append(tmp_state)
        return np.array(traj)


class SamplingBasedKinodynamicPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyKinodynamicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius, config.sst_pruning_radius)

    def plan(self, start, goal, timelimit=1.):
        controls = np.array(self.planner.plan(start, goal, timelimit))

        # TODO more efficient way
        return controls


