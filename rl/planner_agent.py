import os, sys
from collections import OrderedDict

import numpy as np
import torch

from util.misc import save_video
from util.gym import action_size
import time
from motion_planners.sampling_based_planner import SamplingBasedPlanner
from util.logger import logger

class PlannerAgent:
    def __init__(self, config, ac_space, non_limited_idx=None, passive_joint_idx=[], ignored_contacts=[], planner_type=None, goal_bias=0.05, is_simplified=False, simplified_duration=0.1, allow_approximate=False, range_=None):

        self._config = config
        self.planner = SamplingBasedPlanner(config, config._xml_path, action_size(ac_space), non_limited_idx, planner_type=planner_type, passive_joint_idx=passive_joint_idx, ignored_contacts=ignored_contacts, contact_threshold=config.contact_threshold, goal_bias=goal_bias, allow_approximate=allow_approximate, is_simplified=is_simplified, simplified_duration=simplified_duration, range_=range_)

        self._is_simplified = is_simplified
        self._simplified_duration = simplified_duration
        self._allow_approximate = allow_approximate

    def plan(self, start, goal, timelimit=None, attempts=15):
        config = self._config
        if timelimit is None:
            timelimit = config.timelimit
        traj, states, valid, exact = self.planner.plan(start, goal, timelimit, config.min_path_len+1, attempts=attempts)
        if self._allow_approximate:
            success = valid
        else:
            success = valid and exact

        if success:
            return traj[1:], success, valid, exact
        else:
            return traj, success, valid, exact

    def get_planner_status(self):
        return self.planner.get_planner_status()

    def isValidState(self, state):
        return self.planner.isValidState(state)



