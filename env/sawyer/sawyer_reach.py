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
        di['goal'] = self.sim.data.qpos[self._ref_target_pos_low:self._ref_target_pos_high+1]
        return di
