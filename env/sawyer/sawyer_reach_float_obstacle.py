import re
from collections import OrderedDict

import numpy as np
from gym import spaces
from env.base import BaseEnv
from env.sawyer.sawyer import SawyerEnv

class SawyerReachFloatObstacleEnv(SawyerEnv):
    def __init__(self, **kwargs):
        super().__init__("sawyer_reach_float_obstacle.xml", **kwargs)
        self._get_reference()



    def _get_reference(self):
        super()._get_reference()

        self.ref_target_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in ['target_x', 'target_y', 'target_z']
        ]
        self.ref_target_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in ['target_x', 'target_y', 'target_z']
        ]

        self.target_id = self.sim.model.body_name2id("target")

    @property
    def init_qpos(self):
        return np.array([0.457, -0.063, 0.0679, 0.12, -0.0666, -0.0258, 0.00214])

    def _reset(self):
        init_qpos = self.init_qpos + np.random.randn(self.init_qpos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = init_qpos
        self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
        if self._kwargs['task_level'] == 'easy':
            init_target_qpos = np.array([0.6, -0.6, 1.2])
            init_target_qpos += np.random.randn(init_target_qpos.shape[0]) * 0.02
            self.sim.data.qpos[self.ref_target_pos_indexes] = init_target_qpos
            self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
            self.sim.forward()
            self.goal = init_target_qpos
        else:
            while True:
                init_target_qpos = np.random.uniform(low=[0.5, -0.6, 0.9], high=[0.8, 0.0, 1.3])
                self.sim.data.qpos[self.ref_target_pos_indexes] = init_target_qpos
                self.sim.data.qvel[self.ref_joint_vel_indexes] = 0.
                self.sim.forward()
                if not self.on_collision('target'):
                    self.goal = init_target_qpos
                    break

        return self._get_obs()

    def compute_reward(self, action):
        info = {}
        reward = 0

        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        target_pos = self.sim.data.qpos[self.ref_target_pos_indexes]
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
        di['target'] = self.sim.data.qpos[self.ref_target_pos_indexes]
        return di

    @property
    def static_geoms(self):
        return ['obstacle1_geom', 'obstacle2_geom']

    @property
    def static_geom_ids(self):
        #  table_collision, obstacle1~4
        return [self.sim.model.geom_name2id(name) for name in self.static_geoms]

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
            self.sim.data.qfrc_applied[
                self.ref_target_vel_indexes
            ] = self.sim.data.qfrc_bias[
                self.ref_target_vel_indexes
            ]

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
