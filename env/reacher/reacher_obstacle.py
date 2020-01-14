import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv


class ReacherObstacleEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, reward_type='dense', goal_threshold=0.03, **kwarg):
        super().__init__("reacher_obstacle.xml")
        self.obstacle_names = list(filter(lambda x: re.search(r'obstacle', x), self.model.body_names))
        self._env_config['reward_type'] = reward_type
        self._env_config['distance_threshold'] = goal_threshold

    def _reset(self):
        self._set_camera_position(0, [0, -1.0, 1.0])
        self._set_camera_rotation(0, [0, 0, 0])
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.sim.data.qpos.ravel()
        while True:
            self.goal = np.random.uniform(low=-.3, high=.3, size=2)
            # not too close and far from the root
            if np.linalg.norm(self.goal) > 0.15:
                break
        qpos[-2:] = self.goal
        qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obstacle_states(self):
        obstacle_states = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
        return np.array(obstacle_states)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self._get_obstacle_states(),
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
        done, info, penalty = self._after_step(reward, done, info)
        return obs, reward, done, info
