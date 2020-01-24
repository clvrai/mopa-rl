import os, sys

import numpy as np
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyKinematicPlanner


class SamplingBasedPlanner:
    def __init__(self, config, xml_path, num_actions):
        self.config = config
        self.planner = PyKinematicPlanner(xml_path.encode('utf-8'), config.planner_type.encode('utf-8'), num_actions,
                                 config.sst_selection_radius, config.sst_pruning_radius,
                                 config.planner_objective.encode('utf-8'),
                                 config.threshold,
                                 config.range)

    def plan(self, start, goal, timelimit=1., is_clear=False):
        states = np.array(self.planner.plan(start, goal, timelimit, is_clear))
        return states


