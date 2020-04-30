import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from itertools import combinations


class SimpleMoverEnv(BaseEnv):
    """ Mover with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("simple_mover.xml", **kwargs)
        self._env_config.update({
            'subgoal_reward': kwargs['subgoal_reward'],
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

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.right_finger_geoms
        ]


        num_actions = self.dof
        is_limited = np.array([True] * self.dof)
        minimum = np.ones(self.dof) * -1.
        maximum = np.ones(self.dof) * 1
        self._ac_rescale = 0.5

        self._minimum = minimum
        self._maximum = maximum
        self.action_space = spaces.Dict([
            ('default', spaces.Box(low=minimum, high=maximum, dtype=np.float32))
        ])

        subgoal_minimum = np.ones(len(self.ref_joint_pos_indexes)) * -1.
        subgoal_maximum = np.ones(len(self.ref_joint_pos_indexes)) * 1
        self.subgoal_space = spaces.Dict([
            ('default', spaces.Box(low=subgoal_minimum, high=subgoal_maximum, dtype=np.float32))
        ])

        jnt_range = self.sim.model.jnt_range[:num_actions]
        is_jnt_limited = self.sim.model.jnt_limited[:num_actions].astype(np.bool)
        jnt_minimum = np.full(num_actions, fill_value=-np.inf, dtype=np.float)
        jnt_maximum = np.full(num_actions, fill_value=np.inf, dtype=np.float)
        jnt_minimum[is_jnt_limited], jnt_maximum[is_jnt_limited] = jnt_range[is_jnt_limited].T
        jnt_minimum[np.invert(is_jnt_limited)] = -3.14
        jnt_maximum[np.invert(is_jnt_limited)] = 3.14
        self._is_jnt_limited = is_jnt_limited
        self._jnt_minimum = jnt_minimum
        self._jnt_maximum = jnt_maximum

        self.joint_space = spaces.Dict([
            ('default', spaces.Box(low=jnt_minimum, high=jnt_maximum, dtype=np.float32))
        ])

        self._primitive_skills = kwargs['primitive_skills']
        if len(self._primitive_skills) != 3:
            self._primitive_skills = ['reach', 'grasp', 'manipulation']
        self._num_primitives = len(self._primitive_skills)
        self._env_debug = kwargs['env_debug']

        assert self._num_primitives == 3


    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        self._stages = [False] * self._num_primitives
        self._stage = 0
        while True:
            if self._env_debug:
                goal = np.random.uniform(low=0, high=0.2, size=2)
                box = np.random.uniform(low=-0.2, high=0., size=2)
            else:
                goal = np.random.uniform(low=-0.2, high=0.2, size=2)
                box = np.random.uniform(low=-0.2, high=0.2, size=2)
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq) + self.sim.data.qpos.ravel()
            qpos[3] = 0.
            qpos[4] = 0.
            qpos[-4:-2] = goal
            qpos[-2:] = box
            qvel = np.random.uniform(low=-.005, high=.005, size=self.sim.model.nv) + self.sim.data.qvel.ravel()
            qvel[len(self.ref_joint_vel_indexes)+1] = 0.
            qvel[len(self.ref_joint_vel_indexes)+2] = 0.
            qvel[-4:-2] = 0
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if self.sim.data.ncon == 0 and np.linalg.norm(goal) > 0.2 and self._get_distance('box', 'target') > 0.1 and \
                    self._get_distance('fingertip', 'box') > 0.1 and np.linalg.norm(box) > 0.1: #make the task harder
                self.goal = goal
                self.box = box
                break
        return self._get_obs()

    def initialize_joints(self):
        while True:
            qpos = np.random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq) + self.sim.data.qpos.ravel()
            qpos[-4:-2] = self.goal
            qpos[-2:] = self.box
            self.set_state(qpos, self.sim.data.qvel.ravel())
            if self.sim.data.ncon == 0:
                break
    @property
    def gripper_geoms(self):
        return self.left_finger_geoms + self.right_finger_geoms

    @property
    def left_finger_geoms(self):
        return ["l_finger_g0"]

    @property
    def right_finger_geoms(self):
        return ["r_finger_g0"]

    @property
    def body_geoms(self):
        return ['root', 'link0', 'link1', 'link2', 'gripper_base_geom']

    @property
    def agent_geoms(self):
        return self.body_geoms + self.left_finger_geoms + self.right_finger_geoms

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[self.ref_joint_pos_indexes]
        return OrderedDict([
            ('default', np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[-2:],
                self.sim.data.qvel.flat[self.ref_joint_vel_indexes],
                self.sim.data.qvel.flat[-2:], # box vel
                self.sim.data.get_site_xpos('grip_site')[:2]
            ])),
            ('gripper', np.concatenate([
                self.sim.data.qpos.flat[len(self.ref_joint_pos_indexes):len(self.ref_joint_pos_indexes)+2],
                self.sim.data.qvel.flat[len(self.ref_joint_vel_indexes):len(self.ref_joint_vel_indexes)+2]
            ])),
            ('goal', self.sim.data.qpos.flat[self.sim.model.nu:-2])
        ])

    def _format_action(self, action):
        # format action for gripper
        assert len(action) == 1
        return np.array([1*action[0], -1*action[0]])

    @property
    def dof(self):
        return len(self.joint_names) + 1

    @property
    def observation_space(self):
        return spaces.Dict([
            ('default', spaces.Box(shape=(15,), low=-1, high=1, dtype=np.float32)),
            ('gripper', spaces.Box(shape=(4,), low=-1, high=1, dtype=np.float32)),
            ('goal', spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32))
        ])

    @property
    def get_joint_positions(self):
        """
        The joint position except for goal states
        """
        return self.sim.data.qpos.ravel()[self.ref_joint_pos_indexes]

    def _get_control(self, state, prev_state, target_vel):
        alpha = 0.95
        p_term = self._kp * (state - self.sim.data.qpos[self.ref_joint_pos_indexes])
        d_term = self._kd * (target_vel * 0 - self.sim.data.qvel[self.ref_joint_vel_indexes])
        self._i_term = alpha * self._i_term + self._ki * (prev_state - self.sim.data.qpos[self.ref_joint_pos_indexes])
        action = p_term + d_term + self._i_term

        return action

    def form_action(self, next_qpos, skill=None):
        joint_ac = next_qpos[self.ref_joint_pos_indexes] - self.sim.data.qpos.copy()[self.ref_joint_pos_indexes]
        if skill == 2:
            gripper_ac = np.array([1.])
        else:
            gripper_ac = np.array([0.])
        ac = OrderedDict([('default', np.concatenate([
            joint_ac, gripper_ac
        ]))])
        return ac

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def compute_reward(self, action):
        info = {}
        reward_type = self._env_config['reward_type']
        reward_ctrl = self._ctrl_reward(action)
        if reward_type == 'dense':
            reach_multi = 0.35
            collision_multi = 0.2
            gripper_multi = 0.
            grasp_multi = 0.75
            move_multi = 0.9
            dist_box_to_gripper = np.linalg.norm(self._get_pos('box')-self.sim.data.get_site_xpos('grip_site'))
            reward_reach = (1-np.tanh(5.0*dist_box_to_gripper)) * reach_multi
            reward_gripper = (1-np.tanh(5.0*self._cos_vec(self._get_pos('box'),
                                           self._get_pos('l_finger_g0'),
                                           self._get_pos('r_finger_g0')))) * gripper_multi
            has_grasp = self._has_grasp()
            has_self_collision = self._has_self_collision()
            # reward_grasp = (int(has_grasp) - int(has_self_collision)*0.2*int(has_grasp)) * grasp_multi
            reward_grasp = int(has_grasp) * grasp_multi
            reward_collision = -int(has_self_collision) * collision_multi
            # reward_grasp = int(has_grasp) * grasp_multi
            reward_move = (1-np.tanh(5.0*self._get_distance('box', 'target'))) * move_multi * int(self._has_grasp())
            reward_ctrl = self._ctrl_reward(action)

            reward = reward_reach + reward_gripper + reward_grasp + reward_move + reward_ctrl + reward_collision
            # reward = max((reward_reach, reward_grasp, reward_move)) + reward_collision + reward_ctrl

            info = dict(reward_reach=reward_reach, reward_gripper=reward_gripper,
                        reward_grasp=reward_grasp, reward_move=reward_move,
                        reward_collision=reward_collision, reward_ctrl=reward_ctrl)
        else:
            reward = -(self._get_distance('box', 'target') > self._env_config['distance_threshold']).astype(np.float32)

        return reward, info

    def _has_grasp(self):
        touch_left_finger = False
        touch_right_finger = False
        box_geom_id = self.sim.model.geom_name2id('box')
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == box_geom_id:
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 == box_geom_id:
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True

        return touch_left_finger and touch_right_finger


    def check_stage(self):
        dist_box_to_gripper = np.linalg.norm(self._get_pos('box')-self.sim.data.get_site_xpos('grip_site'))
        if dist_box_to_gripper < 0.2:
            self._stages[0] = True
        else:
            self._stages[0] = False

        if self._has_grasp() and self._stages[0]:
            self._stages[1] = True
        else:
            self._stages[1] = False

        if self._get_distance('box', 'target') < self._env_config['distance_threshold'] and self._stages[1]:
            self._stages[2] = True
        else:
            self._stages[2] = False

    def _has_self_collision(self):
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(c.geom1)
            geom2 = self.sim.model.geom_id2name(c.geom2)
            if geom1 in self.agent_geoms and geom2 in self.agent_geoms:
                return True
        return False


    def _step(self, action, is_planner):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        done = False
        if not is_planner or self._prev_state is None:
            self._prev_state = self.sim.data.qpos[self.ref_joint_pos_indexes].copy()

        if not is_planner:
            rescaled_ac = action[:-1] * self._ac_rescale
        else:
            rescaled_ac = action[:-1]
        desired_state = self._prev_state + rescaled_ac # except for gripper action


        n_inner_loop = int(self._frame_dt/self.dt)

        target_vel = (desired_state-self._prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            arm_action = self._get_control(desired_state, self._prev_state, target_vel)
            gripper_action_in = action[len(self.joint_names):len(self.joint_names)+1]
            gripper_action = self._format_action(gripper_action_in)
            ac = np.concatenate((arm_action, gripper_action))
            self._do_simulation(ac)

        reward, info = self.compute_reward(action)
        self.check_stage()

        obs = self._get_obs()
        self._prev_state = np.copy(desired_state)

        if self._get_distance('box', 'target') < self._env_config['distance_threshold'] and self._has_grasp():
            if self._env_config['has_terminal']:
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
