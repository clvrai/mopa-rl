import os
import time
import logging
import traceback
from collections import OrderedDict
from env.robosuite.utils.transform_utils import *

try:
    import mujoco_py
except ImportError as e:
    raise Exception("{}. (need to install mujoco_py)".format(e))

import scipy.misc
import numpy as np
import gym
from gym import spaces, error

import env.transform_utils as T
from util.logger import logger
np.set_printoptions(suppress=True)
from env.base import BaseEnv

class SawyerEnv(BaseEnv):
    def __init__(self, xml_path, **kwargs):
        super().__init__(xml_path, **kwargs)
        # Action
        num_actions = self.dof
        is_limited = np.array([True] * self.dof)
        minimum = -np.ones(self.dof)
        maximum = np.ones(self.dof)
        self._ac_scale = 0.1

        self._minimum = minimum
        self._maximum = maximum
        self.action_space = spaces.Dict([
            ('default', spaces.Box(low=minimum, high=maximum, dtype=np.float32))
        ])

        jnt_range = self.sim.model.jnt_range
        is_jnt_limited = self.sim.model.jnt_limited.astype(np.bool)
        jnt_minimum = np.full(len(is_jnt_limited), fill_value=-np.inf, dtype=np.float)
        jnt_maximum = np.full(len(is_jnt_limited), fill_value=np.inf, dtype=np.float)
        jnt_minimum[is_jnt_limited], jnt_maximum[is_jnt_limited] = jnt_range[is_jnt_limited].T
        jnt_minimum[np.invert(is_jnt_limited)] = -3.14
        jnt_maximum[np.invert(is_jnt_limited)] = 3.14
        self.use_robot_indicator = kwargs["use_robot_indicator"]
        self.use_target_robot_indicator = kwargs["use_target_robot_indicator"]
        self._is_jnt_limited = is_jnt_limited

        self._prev_state = None
        self._i_term = None

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    @property
    def observation_space(self):
        observation = self._get_obs()
        observation_space = OrderedDict()
        for k, v in observation.items():
            observation_space[k] = spaces.Box(low=-1., high=1, shape=v.shape)
        observation_space = spaces.Dict(observation_space)
        return observation_space

    @property
    def robot_joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def robot_bodies(self):
        return ["right_l{}".format(x) for x in range(7)]

    @property
    def robot_indicator_joints(self):
        return ["right_j{}_indicator".format(x) for x in range(7)]

    @property
    def robot_indicator_bodies(self):
        return ["right_l{}_indicator".format(x) for x in range(7)]

    @property
    def target_robot_indicator_joints(self):
        return ["right_j{}_target".format(x) for x in range(7)]

    @property
    def target_robot_indicator_bodies(self):
        return ["right_l{}_target".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])

    @property
    def gripper_joints(self):
        return ["rc_close", "lc_close"]

    @property
    def gripper_init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def dof(self):
        return 8

    @property
    def robot_dof(eslf):
        return 7

    @property
    def manipulation_geom(self):
        return []

    @property
    def manipulation_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.manipulation_geom]

    def _get_reference(self):
        super()._get_reference()

        if self.use_robot_indicator:
            self.ref_indicator_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.robot_indicator_joints
            ]
            self.ref_indicator_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.robot_indicator_joints
            ]

        if self.use_target_robot_indicator:
            self.ref_target_indicator_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.target_robot_indicator_joints
            ]
            self.ref_target_indicator_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.target_robot_indicator_joints
            ]

        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]


        # indices for grippers in qpos, qvel
        self.ref_gripper_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
        ]
        self.ref_gripper_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
        ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        self._ref_joint_gripper_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("gripper")
        ]

        # # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        # self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def visualize_goal_indicator(self, qpos):
        if self.use_target_robot_indicator:
            self.sim.data.qpos[self.ref_target_indicator_joint_pos_indexes] = qpos
            for idx in self.target_indicator_agent_geom_ids:
                color = self.sim.model.geom_rgba[idx]
                color[-1] = 0.3
                self.sim.model.geom_rgba[idx] = color
            self.sim.forward()

    def visualize_dummy_indicator(self, qpos):
        if self.use_robot_indicator:
            self.sim.data.qpos[self.ref_indicator_joint_pos_indexes] = qpos
            for idx in self.indicator_agent_geom_ids:
                color = self.sim.model.geom_rgba[idx]
                color[-1] = 0.3
                self.sim.model.geom_rgba[idx] = color
            self.sim.forward()

    def reset_visualized_indicator(self):
        for idx in self.indicator_agent_geom_ids + self.target_indicator_agent_geom_ids:
            color = self.sim.model.geom_rgba[idx]
            color[-1] = 0.
            self.sim.model.geom_rgba[idx] = color

    @property
    def agent_geom_ids(self):
        body_ids = []
        for body_name in self.robot_bodies:
            body_ids.append(self.sim.model.body_name2id(body_name))

        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return geom_ids

    @property
    def target_indicator_agent_geom_ids(self):
        if self.use_target_robot_indicator:
            body_ids = []
            for body_name in self.target_robot_indicator_bodies:
                body_ids.append(self.sim.model.body_name2id(body_name))

            geom_ids = []
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                if body_id in body_ids:
                    geom_ids.append(geom_id)
            return geom_ids
        else:
            return []

    @property
    def indicator_agent_geom_ids(self):
        if self.use_robot_indicator:
            body_ids = []
            for body_name in self.robot_indicator_bodies:
                body_ids.append(self.sim.model.body_name2id(body_name))

            geom_ids = []
            for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
                if body_id in body_ids:
                    geom_ids.append(geom_id)
            return geom_ids
        else:
            return []

    def form_action(self, next_qpos, curr_qpos=None):
        if curr_qpos is None:
            curr_qpos = self.sim.data.qpos.copy()
        joint_ac = next_qpos[self.ref_joint_pos_indexes] - curr_qpos[self.ref_joint_pos_indexes]
        ac = OrderedDict([('default', np.concatenate([joint_ac, [0.]]))])
        return ac

    def compute_reward(self, action):
        pass

    def _get_obs(self):
        di = super()._get_obs()
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self.ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self.ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        di["gripper_qpos"] = np.array(
            [self.sim.data.qpos[x] for x in self.ref_gripper_joint_pos_indexes]
        )
        di["gripper_qvel"] = np.array(
            [self.sim.data.qvel[x] for x in self.ref_gripper_joint_vel_indexes]
        )

        di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
        di["eef_quat"] = convert_quat(
            self.sim.data.get_body_xquat("right_ee_attchment"), to="xyzw"
        )

        # add in gripper information
        robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])

        di["robot-state"] = np.concatenate(robot_states)
        return di

    def _gripper_format_action(self, gripper_ac):
        gripper_state = self.sim.data.qpos[self.ref_gripper_joint_pos_indexes]
        gripper_ac *= 0.15
        return gripper_state + gripper_ac

    def _step(self, action, is_planner=False):
        """
        (Optional) does gripper visualization after actions.
        """
        assert len(action) == self.dof, "environment got invalid action dimension"

        if not is_planner or self._prev_state is None:
            self._prev_state = self.sim.data.qpos[self.ref_joint_pos_indexes].copy()

        if self._i_term is None:
            self._i_term = np.zeros_like(self.mujoco_robot.dof)

        if is_planner:
            rescaled_ac = action[:self.robot_dof]
        else:
            rescaled_ac = action[:self.robot_dof] * self._ac_scale
        desired_state = self._prev_state + rescaled_ac
        arm_action = desired_state
        gripper_action = self._gripper_format_action(np.array([action[-1]]))
        converted_action = np.concatenate([arm_action, gripper_action])

        n_inner_loop = int(self._frame_dt/self.dt)
        for _ in range(n_inner_loop):
            self.sim.data.qfrc_applied[self.ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self.ref_joint_vel_indexes].copy()
            if self.use_robot_indicator:
                self.sim.data.qfrc_applied[
                    self.ref_indicator_joint_pos_indexes
                ] = self.sim.data.qfrc_bias[
                    self.ref_indicator_joint_pos_indexes
                ].copy()

            if self.use_target_robot_indicator:
                self.sim.data.qfrc_applied[
                    self.ref_target_indicator_joint_pos_indexes
                ] = self.sim.data.qfrc_bias[
                    self.ref_target_indicator_joint_pos_indexes
                ].copy()
            self._do_simulation(converted_action)

        self._prev_state = np.copy(desired_state)
        reward, info = self.compute_reward(action)

        return self._get_obs(), reward, self._terminal, info
