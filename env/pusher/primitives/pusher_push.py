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
            'success_reward': 10
            #'success_reward': 30
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
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.1 and \
                    self._get_distance('box', 'fingertip') < 0.05 and \
                    self._get_distance('box', 'target') < self._get_distance('fingertip', 'target') and \
                    np.linalg.norm(box) > 0.1 and \
                    self._get_distance('box', 'target') > self._env_config['distance_threshold']:
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

        reward_type = self._env_config['reward_type']
        reward_ctrl = self._ctrl_reward(action)
        if reward_type == 'dense':
            reward_dist = -self._get_distance("box", "target")
            reward = reward_dist + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        elif reward_type == 'dist_diff':
            pre_reward_dist = self._get_distance("box", "target")
        elif reward_type == 'inverse':
            reward_0 = 100.
            reward_inv_dist = reward_0 / (self._get_distance('box', 'target')+1.)
            reward = reward_inv_dist + reward_ctrl
            info = dict(reward_inv=reward_inv_dist, reward_ctrl=reward_ctrl)
        elif reward_type == 'exp':
            reward_exp_dist = np.exp(-self._get_distance('box', 'target'))
            reward = reward_exp_dist + reward_ctrl
            info = dict(reward_exp_dist=reward_exp_dist, reward_ctrl=reward_ctrl)
        elif reward_type == 'composition':
            reward_dist = -self._get_distance("box", "target")
            reward_near = -self._get_distance("fingertip", "box")
            reward_ctrl = self._ctrl_reward(action)
            reward = reward_dist + 0.5*reward_near + reward_ctrl
            info = dict(reward_dist=reward_dist, reward_near=reward_near, reward_ctrl=reward_ctrl)
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
            reward_dist_diff = pre_reward_dist - post_reward_dist
            info = dict(reward_dist_diff=reward_dist_diff, reward_ctrl=reward_ctrl)
            reward = reward_dist_diff + reward_ctrl

        if self._get_distance('box', 'target') < self._env_config['distance_threshold']:
            done = True
            self._success = True
        return obs, reward, done, info

