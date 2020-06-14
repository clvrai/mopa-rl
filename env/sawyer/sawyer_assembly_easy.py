import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from env.sawyer.sawyer import SawyerEnv

class SawyerAssemblyEasyEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super().__init__("sawyer_assembly_easy.xml", **kwargs)
        self._get_reference()



    def _get_reference(self):
        super()._get_reference()

    def _reset(self):
        init_qpos = self.init_qpos + np.random.randn(self.init_qpos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
        self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        # if self._kwargs['task_level'] == 'easy':
        #     init_target_qpos = np.array([0.6, 0.0, 1.2])
        #     init_target_qpos += np.random.randn(init_target_qpos.shape[0]) * 0.02
        # else:
        #     init_target_qpos = np.random.uniform(low=[0.5, -0.3, 0.9], high=[0.8, 0.3, 1.3])
        #
        # self.goal = init_target_qpos
        # self.sim.data.qpos[self.ref_target_pos_indexes] = self.goal
        # self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        self.sim.forward()

        return self._get_obs()

    def compute_reward(self, action):
        info = {}
        reward = 0
        peg_pos = self._get_pos("peg1")

        nut_pos = self._get_pos("SquareNut0")
        dist = np.linalg.norm(peg_pos[:2] - nut_pos[:2])
        reward_reach = -dist


        info = dict(reward_reach=reward_reach)
        if self.on_peg(peg_pos):
            reward += 1.0
            self._success = True
            self._terminal = True

        return reward, info

    def on_peg(self, peg_pos):
        res = False
        nut_pos = self._get_pos("SquareNut0")

        if (
            abs(nut_pos[0] - peg_pos[0]) < 0.03
            and abs(nut_pos[1] - peg_pos[1]) < 0.03
            and nut_pos[2] < self.sim.data.get_site_xpos("table_top") + 0.05
        ):
            res = True
        return res

    def _get_obs(self):
        di = super()._get_obs()
        # di['target'] = self.sim.data.qpos[self.ref_target_pos_indexes]
        return di

    @property
    def static_geom_ids(self):
        return []

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
            self._do_simulation(desired_state)

        self._prev_state = np.copy(desired_state)
        reward, info = self.compute_reward(action)

        return self._get_obs(), reward, self._terminal, info
