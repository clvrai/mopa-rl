import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from env.sawyer.sawyer import SawyerEnv

class SawyerReachEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super().__init__("sawyer_reach.xml", **kwargs)
        self._get_reference()



    def _get_reference(self):
        super()._get_reference()

        ind_qpos = (self.sim.model.get_joint_qpos_addr("target_x"), self.sim.model.get_joint_qpos_addr('target_z'))
        self._ref_target_pos_low, self._ref_target_pos_high = ind_qpos

        ind_qvel = (self.sim.model.get_joint_qvel_addr("target_x"), self.sim.model.get_joint_qvel_addr('target_z'))
        self._ref_target_vel_low, self._ref_target_vel_high = ind_qvel

        self.target_id = self.sim.model.body_name2id("target")


    def compute_reward(self, action):
        info = {}
        reward = 0

        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        target_pos = self.sim.data.qpos[self._ref_target_pos_low:self._ref_target_pos_high+1]
        dist = np.linalg.norm(target_pos-gripper_site_pos)
        reward_reach = -dist
        reward += reward_reach
        info = dict(reward_reach=reward_reach)
        if dist < self._kwargs['distance_threshold']:
            # reward += 1.0
            self._success = True
            self._terminal = True

        return reward, info

    def _get_obs(self):
        di = super()._get_obs()
        di['target'] = self.sim.data.qpos[self._ref_target_pos_low:self._ref_target_pos_high+1]
        return di

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
