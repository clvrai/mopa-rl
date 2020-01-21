import os, sys

import numpy as np
import shuntil
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyPlanner

lock = Lock()
n = 0

class SamplingBasedPlanner:
    def __init__(self, config, xml_path):
        self.config
        self.planner = PyPlanner(xml_path)

    def plan(self, start, goal, planner='sst', timelimit=1.):
       e traj = np.array(self.planner.planning(start, goal, timelimit))
        return traj


