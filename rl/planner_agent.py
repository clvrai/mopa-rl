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
    def __init__(self, config, ac_space, non_limited_idx=None, passive_joint_idx=[], ignored_contacts=[]):

        self._config = config
        self.planner = SamplingBasedPlanner(config, config._xml_path, action_size(ac_space), non_limited_idx, passive_joint_idx=passive_joint_idx, ignored_contacts=ignored_contacts)

    def plan(self, start, goal):
        config = self._config
        traj, states, success = self.planner.plan(start, goal, config.timelimit, config.min_path_len+1)
        if success:
            return traj[1:], success
        else:
            return traj, success

    def remove_collision(self, geom, contype=0, conaffinity=0):
        self.planner.remove_collision(geom, contype, conaffinity)
        logger.info('change (%s): contype (%d) conaffinity (%d)', geom, contype, conaffinity)





