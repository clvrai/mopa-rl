import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from env.sawyer.sawyer import SawyerEnv
from env.robosuite.utils.transform_utils import *

class SawyerPickPlaceEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super().__init__("sawyer_pick_place.xml", **kwargs)
        self._get_reference()

    @property
    def init_qpos(self):
        return np.array([-0.0305, -0.7325, 0.03043, 1.16124, 1.87488, 0, 0])

    def _get_reference(self):
        super()._get_reference()

        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.cube_geom_id = self.sim.model.geom_name2id("cube")
        self.cube_site_id = self.sim.model.site_name2id("cube")


    def _reset(self):
        init_qpos = self.init_qpos + np.random.randn(self.init_qpos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
        self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        self.sim.forward()

        return self._get_obs()


    def initialize_joints(self):
        init_qpos = self.init_qpos + np.random.randn(self.init_qpos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
        self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        self.sim.forward()

    @property
    def left_finger_geoms(self):
        return ["l_finger_g0", "l_finger_g1", "l_fingertip_g0"]

    @property
    def right_finger_geoms(self):
        return ["r_finger_g0", "r_finger_g1", "r_fingertip_g0"]

    @property
    def l_finger_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.left_finger_geoms]

    @property
    def r_finger_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.right_finger_geoms]

    @property
    def gripper_bodies(self):
        return ["clawGripper", "rightclaw", 'leftclaw',
                'right_gripper_base', 'right_gripper', 'r_gripper_l_finger_tip', 'r_gripper_r_finger_tip']
    @property
    def gripper_indicator_bodies(self):
        return ["clawGripper_indicator", "rightclaw_indicator", 'leftclaw_indicator',
                'right_gripper_base_indicator', 'r_gripper_l_finger_tip_indicator', 'r_gripper_r_finger_tip_indicator']
    @property
    def gripper_target_bodies(self):
        return ["clawGripper_target", "rightclaw_target", 'leftclaw_target',
                'right_gripper_base_target', 'r_gripper_l_finger_tip_target', 'r_gripper_r_finger_tip_target']


    def compute_reward(self, action):
        reward_type = self._kwargs['reward_type']
        info = {}
        reward = 0

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        reward_reach = 0.
        gripper_site_pos = self.sim.data.get_site_xpos("grip_site")
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        gripper_to_cube = np.linalg.norm(cube_pos-gripper_site_pos)
        reward_reach = (1-np.tanh(10*gripper_to_cube)) * reach_mult

        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.cube_geom_id:
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 == self.cube_geom_id:
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_right_finger and touch_left_finger
        reward_grasp = int(has_grasp) * grasp_mult

        reward_lift = 0.
        object_z_locs = self.sim.data.body_xpos[self.cube_body_id][2]
        if reward_grasp > 0.:
            z_target = self._get_pos("bin1")[2] + 0.25
            z_dist = np.maximum(z_target-object_z_locs, 0.)
            reward_lift = grasp_mult + (1-np.tanh(15*z_dist)) * (lift_mult-grasp_mult)

        reward_hover = 0.
        target_bin = self._get_pos('bin2')
        object_xy_locs = self.sim.data.body_xpos[self.cube_body_id][:2]
        y_check = (
            np.abs(object_xy_locs[1]-(target_bin[1]-0.075)) < 0.075
        )
        x_check = (
            np.abs(object_xy_locs[0]-(target_bin[0]-0.075)) < 0.075
        )
        object_above_bin = np.logical_and(x_check, y_check)
        object_not_above_bin = np.logical_not(object_above_bin)
        dist = np.linalg.norm(target_bin[:2]-object_xy_locs)
        reward_hover += int(object_above_bin) * (lift_mult + (1-np.tanh(10*dist)) * (hover_mult - lift_mult))
        reward_hover += int(object_not_above_bin) * (reward_lift + (1-np.tanh(10*dist))*(hover_mult-lift_mult))

        reward += max(reward_reach, reward_grasp, reward_lift, reward_hover)
        info = dict(reward_reach=reward_reach, reward_grasp=reward_grasp,
                    reward_lift=reward_lift, reward_hover=reward_hover)

        if object_above_bin and object_z_locs < self._get_pos('bin1')[2] + 0.1:
            reward += self._kwargs['success_reward']
            self._success = True
        else:
            self._success = False

        return reward, info

    def _get_obs(self):
        di = super()._get_obs()
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        cube_quat = convert_quat(
            np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
        )
        di["cube_pos"] = cube_pos
        di["cube_quat"] = cube_quat
        gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
        di["gripper_to_cube"] = gripper_site_pos - cube_pos


        return di

    @property
    def static_bodies(self):
        return ['table', 'bin1', 'bin2']

    @property
    def static_geoms(self):
        return []

    @property
    def static_geom_ids(self):
        body_ids = []
        for body_name in self.static_bodies:
            body_ids.append(self.sim.model.body_name2id(body_name))

        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return geom_ids

    @property
    def manipulation_geom(self):
        return ['cube']

    @property
    def manipulation_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.manipulation_geom]


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
            rescaled_ac = np.clip(action[:self.robot_dof], -self._ac_scale, self._ac_scale)
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
