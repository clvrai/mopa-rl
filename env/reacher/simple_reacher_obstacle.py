import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv

class SimpleReacherObstacleEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("simple_reacher_obstacle.xml", **kwargs)
        self.obstacle_names = list(filter(lambda x: re.search(r'obstacle', x), self.sim.model.body_names))
        self._env_config.update({
            'success_reward': kwargs['success_reward']
        })
        self.joint_names = ["joint0", "joint1", "joint2"]
        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.joint_names
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.joint_names
        ]
        self._ac_rescale = 0.5
        self._subgoal_scale = kwargs['subgoal_scale']
        subgoal_minimum = np.ones(len(self.ref_joint_pos_indexes)) * -self._subgoal_scale
        subgoal_maximum = np.ones(len(self.ref_joint_pos_indexes)) * self._subgoal_scale
        self.subgoal_space = spaces.Dict([
            ('default', spaces.Box(low=subgoal_minimum, high=subgoal_maximum, dtype=np.float32))
        ])

        self._primitive_skills = kwargs['primitive_skills']
        if len(self._primitive_skills) != 1:
            self._primitive_skills = ['reach']
        self._num_primitives = len(self._primitive_skills)

    def _reset(self):
        self._stages = [False] * self._num_primitives
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = np.random.uniform(low=[-0.2, 0.1], high=[0., 0.2], size=2)
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq) + self._init_qpos
            qpos[self.sim.model.nu:] = goal
            qvel = np.random.uniform(low=-.005, high=.005, size=self.sim.model.nv) + self._init_qvel
            qvel[self.sim.model.nu:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2:
                #and self._is_far_from_obstacle: # might need to take action for one step to check the collision sim step.
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

    def _get_obstacle_states(self):
        obstacle_states = []
        obstacle_size = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
            obstacle_size.extend(self._get_size(name)[:2])
        return np.concatenate([obstacle_states, obstacle_size])

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:self.sim.model.nu]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[self.sim.model.nu:],
                self.sim.data.qvel.flat[:self.sim.model.nu],
            ]))
        ])

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(11, ), low=-1, high=1, dtype=np.float32))
        ])

    @property
    def get_joint_positions(self):
        """
        The joint position except for goal states
        """
        return self.sim.data.qpos.ravel()[:self.sim.model.nu]

    def check_stage(self):
        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold'] and self._stages[0]:
            self._stages[0] = True
        else:
            self._stages[0] = False

    def compute_reward(self, action):
        info = {}
        reward_type = self._env_config['reward_type']
        reward_ctrl = self._ctrl_reward(action)
        if reward_type == 'dense':
            reward_reach = (1-np.tanh(5.0*self._get_distance('fingertip', 'target')))
            reward_ctrl = self._ctrl_reward(action)

            reward = reward_reach + reward_ctrl

            info = dict(reward_reach=reward_reach, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('box', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        return reward, info

    def _step(self, action, is_planner=False):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        info = {}
        done = False
        if not is_planner or self._prev_state is None:
            self._prev_state = self.get_joint_positions

        if not is_planner:
            rescaled_ac = action * self._ac_rescale
        else:
            rescaled_ac = action
        desired_state = self._prev_state + rescaled_ac # except for gripper action

        n_inner_loop = int(self._frame_dt/self.dt)
        reward, info = self.compute_reward(action)
        self.check_stage()

        target_vel = (desired_state-self._prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            action = self._get_control(desired_state, self._prev_state, target_vel)
            self._do_simulation(action)

        obs = self._get_obs()
        self._prev_state = np.copy(desired_state)

        if self._get_distance('fingertip', 'target') < self._env_config['distance_threshold']:
            self._success = True
            # done = True
            if self._kwargs['has_terminal']:
                done = True
                self._success = True
            else:
                if self._episode_length == self._env_config['max_episode_steps']-1:
                    self._success = True
            reward += self._env_config['success_reward']
        return obs, reward, done, info

    def compute_subgoal_reward(self, name, info):
        reward_subgoal_dist = -0.5*self._get_distance(name, "subgoal")
        info['reward_subgoal_dist'] = reward_subgoal_dist
        return reward_subgoal_dist, info

    def get_next_primitive(self, prev_primitive):
        for i in reversed(range(self._num_primitives)):
            if self._stages[i]:
                if i == self._num_primitives-1:
                    return self._primitive_skills[i]
                else:
                    return self._primitive_skills[i+1]
        return self._primitive_skills[0]

    def isValidState(self, ignored_contacts=[]):
        if len(ignored_contacts) == 0:
            return self.sim.data.ncon == 0
        else:
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                geom1 = self.sim.model.geom_id2name(c.geom1)
                geom2 = self.sim.model.geom_id2name(c.geom2)
                for pair in ignored_contacts:
                    if geom1 not in pair and geom2 not in pair:
                        return False
            return True


