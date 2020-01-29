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

    def plan(self, start, goal, timelimit=1., is_simplified=False, simplified_duration=1.0):
        states = np.array(self.planner.plan(start, goal, timelimit, is_simplified, simplified_duration))
        actions = []

        # TODO more efficient way
        for i, state in enumerate(states[1:]):
            actions.append((state-states[i])[:-2])
        return states, actions


class SamplingBasedKinodynamicPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyKinodynamicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius, config.sst_pruning_radius)

    def plan(self, start, goal, timelimit=1.):
        controls = np.array(self.planner.plan(start, goal, timelimit))

        # TODO more efficient way
        return controls


