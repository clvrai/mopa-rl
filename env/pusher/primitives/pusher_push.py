import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv
from env.pusher.simple_pusher import SimplePusherEnv

class PusherPushEnv(SimplePusherEnv):
    """ Pusher push primitive environment. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = np.random.uniform(low=-0.2, high=0.2, size=2)
            box = np.random.uniform(low=-0.2, high=0.2, size=2)
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-4:-2] = goal
            qpos[-2:] = box
            qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
            qvel[-4:-2] = 0
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2 and \
                    self._get_distance('box', 'target') < 0.2 and \
                    self._get_distance('box', 'fingertip') < 0.2 and \
                    self._get_distance('target', 'fingertip') > 0.2 and \
                    np.linalg.norm(box) > 0.1:
                self.goal = goal
                break
        return self._get_obs()

