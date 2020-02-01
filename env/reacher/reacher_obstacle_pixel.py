import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from skimage import color, transform


class ReacherObstaclePixelEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("reacher_obstacle.xml", **kwargs)
        self.obstacle_names = list(filter(lambda x: re.search(r'obstacle', x), self.model.body_names))
        self.memory = np.zeros((84, 84, 4))

    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])

        while True:
            goal = np.random.uniform(low=-.4, high=.4, size=2)
            qpos = np.random.uniform(low=-1, high=1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-2:] = goal
            qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2:
                self.goal = goal
                break
        return self._get_obs()

    def initalize_joints(self):
        while True:
            qpos = np.random.uniform(low=-1, high=1, size=self.model.nq) + self.sim.data.qpos.ravel()
            qpos[-2:] = self.goal
            self.set_state(qpos, self.sim.data.qvel.ravel())
            if self.sim.data.ncon == 0:
                break

    def _get_obstacle_states(self):
        obstacle_states = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
        return np.array(obstacle_states)

    def _get_obs(self):
        img = self.sim.render(camera_name=self._camera_name,
                              width=100,
                              height=100,
                              depth=False)
        img = np.flipud(img)
        gray = color.rgb2gray(img)
        gray_resized = transform.resize(gray, (self._img_height, self._img_width)) / 255.
        self.memory[:, :, 1:] = self.memory[:, :, 0:3]
        self.memory[:, :, 0] = gray_resized
        return OrderedDict([('default', self.memory.transpose((2, 0, 1)))])

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(4, self._img_height, self._img_width), low=0, high=255, dtype=np.float32)),
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

        velocity = action * 2 # According to robosuite
        for i in range(self._action_repeat):
            self._do_simulation(velocity)
            if i + 1 < self._action_repeat:
                velocity = self._get_current_error(self.sim.data.qpos.ravel()[:-2], desired_states) * 2

        obs = self._get_obs()
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            done =True
            self._success = True
        return obs, reward, done, info

