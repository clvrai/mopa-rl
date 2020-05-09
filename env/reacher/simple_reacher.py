import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv


class SimpleReacherEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("simple_reacher.xml", **kwargs)
        self._env_config.update({
            'success_reward': kwargs['success_reward'],
            'has_terminal': kwargs['has_terminal']
        })
        self.joint_names = ["joint0", "joint1", "joint2"]
        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.joint_names
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.joint_names
        ]
        self._primitive_skills = kwargs['primitive_skills']
        if len(self._primitive_skills) != 1:
            self._primitive_skills = ['reach']
        self._num_primitives = len(self._primitive_skills)

        subgoal_minimum = np.ones(len(self.ref_joint_pos_indexes)) * -1.
        subgoal_maximum = np.ones(len(self.ref_joint_pos_indexes)) * 1
        self.subgoal_space = spaces.Dict([
            ('default', spaces.Box(low=subgoal_minimum, high=subgoal_maximum, dtype=np.float32))
        ])

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

    def compute_reward(self, action):
        info = {}
        reward_type = self._env_config['reward_type']
        reward_ctrl = self._ctrl_reward(action)
        if reward_type == 'dense':
            reach_multi = 1.
            reward_reach = (1-np.tanh(10.*self._get_distance('fingertip', 'target'))) * reach_multi

            reward = reward_reach + reward_ctrl

            info = dict(reward_reach=reward_reach, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('fingertip', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        return reward, info


    def _step(self, action, is_planner):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        info = {}
        done = False

        desired_state = self.get_joint_positions + action
        reward, info = self.compute_reward(action)

        n_inner_loop = int(self._frame_dt/self.dt)

        prev_state = self.sim.data.qpos[:self.sim.model.nu].copy()
        target_vel = (desired_state-prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            action = self._get_control(desired_state, prev_state, target_vel)
            self._do_simulation(action)

        obs = self._get_obs()
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            if self._env_config['has_terminal']:
                done = True
                self._success = True
            else:
                if self._episode_length == self._env_config['max_episode_steps']-1:
                    self._success = True
            reward += self._env_config['success_reward']
        return obs, reward, done, info

    def get_next_primitive(self, prev_primitive):
        return self._primitive_skills[0]

