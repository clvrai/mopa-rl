import os, sys
from collections import OrderedDict

import numpy as np
import torch

from util.misc import save_video
from util.gym import action_size
import time
from motion_planners.sampling_based_planner import SamplingBasedPlanner

class MpAgent:
    def __init__(self, config, ac_space, non_limited_idx=None):

        self._config = config
        self.planner = SamplingBasedPlanner(config, config._xml_path, action_size(ac_space), non_limited_idx)

    def plan(self, start, goal):
        config = self._config
        traj, states = self.planner.plan(start, goal, config.timelimit, config.max_meta_len)
        return traj

