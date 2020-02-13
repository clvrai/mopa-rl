import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv

class SimpleReacherObstacleEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("simple_reacher_obstacle.xml", **kwargs)
        self.obstacle_names = list(filter(lambda x: re.search(r'obstacle', x), self.model.body_names))

    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = np.random.uniform(low=-.2, high=.2, size=2)
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-2:] = goal
            qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            self.sim.forward()
            self.sim.step()
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2:
                #and self._is_far_from_obstacle: # might need to take action for one step to check the collision sim step.
                self.goal = goal
                break
        return self._get_obs()

    def initalize_joints(self):
        while True:
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-2:] = self.goal
            self.set_state(qpos, self.sim.data.qvel.ravel())
            if self.sim.data.ncon == 0:
                break

    def _get_obstacle_states(self):
        obstacle_states = []
        obstacle_size = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
            obstacle_size.extend(self._get_size(name)[:2])
        return np.concatenate([obstacle_states, obstacle_size])

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self._get_obstacle_states(),
                self._get_pos("target")
            ]))
        ])

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(24,), low=-1, high=1, dtype=np.float32))
        ])

    @property
    def get_joint_positions(self):
        """
        The joint position except for goal states
        """
        return self.sim.data.qpos.ravel()[:-2]

    def _step(self, action):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        info = {}
        done = False
        desired_states = self.get_joint_positions + action

        if self._env_config['reward_type'] == 'dense':
            reward_dist = -self._get_distance("fingertip", "target")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_dist + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('fingertip', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        #velocity = action * 2 # According to robosuite
        velocity = action # According to robosuite
        for i in range(self._action_repeat):
            self._do_simulation(velocity)
            if i + 1 < self._action_repeat:
                velocity = self._get_current_error(self.sim.data.qpos.ravel()[:-2], desired_states)
        obs = self._get_obs()
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            done =True
            self._success = True
        return obs, reward, done, info

    def _kinematics_step(self, states):
        info = {}
        done = False

        if self._env_config['reward_type'] == 'dense':
            reward_dist = -self._get_distance("fingertip", "target")
            reward = reward_dist
            info = dict(reward_dist=reward_dist)
        else:
            reward = -(self._get_distance('fingertip', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        states = np.concatenate((states[:-2], self.goal))
        self.set_state(states, self.sim.data.qvel.ravel())
        obs = self._get_obs()
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            done =True
            self._success = True
        return obs, reward, done, info

