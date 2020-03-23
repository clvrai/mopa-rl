import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv


class SimpleReacherEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("simple_reacher.xml", **kwargs)

    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = np.random.uniform(low=-0.2, high=.2, size=2)
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq) + self.sim.data.qpos.ravel()
            qpos[self.sim.model.nu:] = goal
            qvel = np.random.uniform(low=-.005, high=.005, size=self.sim.model.nv) + self.sim.data.qvel.ravel()
            qvel[self.sim.model.nu:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2:
                self.goal = goal
                break
        return self._get_obs()

    def initalize_joints(self):
        while True:
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq) + self.sim.data.qpos.ravel()
            qpos[self.sim.model.nu:] = self.goal
            self.set_state(qpos, self.sim.data.qvel.ravel())
            if self.sim.data.ncon == 0:
                break

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:self.sim.model.nu]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[self.sim.model.nu:],
                self.sim.data.qvel.flat[:self.sim.model.nu]
            ]))
        ])

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(11,), low=-1, high=1, dtype=np.float32))
        ])


    @property
    def get_joint_positions(self):
        """
        The joint position except for goal states
        """
        return self.sim.data.qpos.ravel()[:self.sim.model.nu]

    def _step(self, action):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        info = {}
        done = False
        desired_state = self.get_joint_positions + action

        if self._env_config['reward_type'] == 'dense':
            reward_dist = -self._get_distance("fingertip", "target")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_dist + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('fingertip', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        n_inner_loop = int(self._frame_dt/self.dt)

        prev_state = self.sim.data.qpos[:self.sim.model.nu].copy()
        target_vel = (desired_state-prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            action = self._get_control(desired_state, prev_state, target_vel)
            self._do_simulation(action)

        obs = self._get_obs()
        # if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
        #     done =True
        #     self._success = True
        return obs, reward, done, info

