import os, sys
from collections import OrderedDict

import numpy as np
import torch

from util.misc import save_video
from util.gym import action_size
import time
from motion_planners.sampling_based_planner import SamplingBasedPlanner

class LowLevelMpAgent:
    def __init__(self, config, ob_space, ac_space):
        if config.planner_type != 'komo':
            self.planner = SamplingBasedPlanner(config, config._xml_path, action_size(ac_space))
        else:
            raise NotImplementedError

        self._config = config
        self._ob_space = ob_space
        self._ac_space = ac_space

    def plan(self, start, goal):
        config = self._config
        traj, actions = self.planner.plan(start, goal, config.timelimit)
        return traj, actions

