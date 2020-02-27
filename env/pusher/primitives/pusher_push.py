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
        self._env_config.update({
            "pos_reward": kwargs['pos_reward_coef'],
        })

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
                    self._get_distance('box', 'target') < self._get_distance('target', 'fingertip') and \
                    self._get_distance('box', 'fingertip') < 0.1 and \
                    self._get_distance('target', 'fingertip') < 0.3 and \
                    self._get_distance('target', 'fingertip') > 0.2 and \
                    self._get_distance('box', 'target') < self._get_distance('target', 'link1') and \
                    np.linalg.norm(box) > 0.1 and self._get_distance('box', 'target') > self._env_config['distance_threshold']:
                self.goal = goal
                break
        return self._get_obs()

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
            reward_dist = -self._env_config['pos_reward'] * self._get_distance("box", "target")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_dist + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        elif self._env_config['reward_type'] == 'dist_diff':
            pre_reward_dist = self._get_distance("box", "target")
            reward_ctrl = self._ctrl_reward(action)
        elif self._env_config['reward_type'] == 'composition':
            reward_box_to_target = -self._box_to_target_coef * self._get_distance("box", "target")
            reward_end_effector_to_box = -self._box_to_target_coef * self._get_distance("end_effector", "box")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_box_to_target + reward_end_effector_to_box + reward_ctrl
            info = dict(reward_box_to_target=reward_box_to_target,
                        reward_end_effector_to_box=reward_end_effector_to_box,
                        reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('box', 'target') > self._env_config['distance_threshold']).astype(np.float32)


        n_inner_loop = int(self._frame_dt/self.dt)

        prev_state = self.sim.data.qpos[:self.model.nu].copy()
        target_vel = (desired_state-prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            action = self._get_control(desired_state, prev_state, target_vel)
            self._do_simulation(action)

        obs = self._get_obs()

        if self._env_config['reward_type'] == 'dist_diff':
            post_reward_dist = self._get_distance("box", "target")
            reward_dist_diff = self._reward_coef * (pre_reward_dist - post_reward_dist)
            info = dict(reward_dist_diff=reward_dist_diff, reward_ctrl=reward_ctrl)
            reward = reward_dist_diff + reward_ctrl

        if self._get_distance('box', 'target') < self._env_config['distance_threshold'] and self._env_config['reward_type'] == 'dense':
            done = True
            self._success = True
        return obs, reward, done, info

