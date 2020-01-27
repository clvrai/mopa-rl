import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv


class ReacherTestEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("reacher_test.xml", **kwargs)

    def _reset(self):
        self._set_camera_position(0, [0, -1.0, 1.0])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = np.random.uniform(low=-.35, high=.35, size=2)
            qpos = np.random.uniform(low=-1, high=1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-2:] = goal
            #qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
            qvel = np.ones(self.model.nv)
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2:
                self.goal = goal
                break
        return self._get_obs()


    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self._get_pos("fingertip") - self._get_pos("target")
            ]))
        ])

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(32,), low=-1, high=1, dtype=np.float32))
        ])

    def _step(self, action):
        info = {}
        done = False
        if self._env_config['reward_type'] == 'dense':
            reward_dist = -self._get_distance("fingertip", "target")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_dist + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('fingertip', 'target') > self._env_config['distance_threshold']).astype(np.float32)
        self._do_simulation(action)
        obs = self._get_obs()
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            done =True
            self._success = True
        return obs, reward, done, info
