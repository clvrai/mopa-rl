from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from env.multitask_env import MultitaskEnv
from env.sawyer.base import SawyerXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

class SawyerNutDisassembleEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=True,
            obs_type='with_goal',
            goal_low=(-0.1, 0.75, 0.17),
            goal_high=(0.1, 0.85, 0.17),
            liftThresh = 0.05,
            rewMode = 'orig',
            rotMode='fixed',
            **kwargs
    ):
        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(0.1, 0.75, 0.02)
        obj_high=(0., 0.85, 0.02)
        SawyerXYZEnv.__init__(
            self,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.8, 0.02]),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low

        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 200
        self.rewMode = rewMode
        self.rotMode = rotMode
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError
        self.reset()

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_assembly_peg.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward , reachRew, reachDist, pickRew, placeRew , placingDist, success = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': placingDist, 'success': success}
        info['goal'] = self.goal
        return ob, reward, done, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        # graspPos =  self.data.get_geom_xpos('RoundNut-8')
        graspPos =  self.get_site_pos('RoundNut-8')
        flat_obs = np.concatenate((hand, graspPos))
        if self.obs_type == 'with_goal_and_id':
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        elif self.obs_type == 'with_goal':
            return np.concatenate([
                    flat_obs,
                    self._state_goal
                ])
        elif self.obs_type == 'plain':
            return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
        else:
            return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        # graspPos =  self.data.get_geom_xpos('RoundNut-8')
        graspPos =  self.get_site_pos('RoundNut-8')
        objPos = self.get_body_com('RoundNut')
        flat_obs = np.concatenate((hand, graspPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('RoundNut-8')
        self.data.site_xpos[self.model.site_name2id('RoundNut')] = (
            objPos
        )
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('pegTop')] = (
            goal[:3]
        )



    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
            self.obj_init_pos = goal_pos[:3]
            # self._state_goal = goal_pos[-3:]
            self._state_goal = goal_pos[:3] + np.array([0, 0, 0.15])
        peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
        self._set_obj_xyz(self.obj_init_pos)
        self._set_goal_marker(self._state_goal)
        self.objHeight = self.data.get_geom_xpos('RoundNut-8')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode = 'general'):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        graspPos = obs[3:6]
        objPos = graspPos
        
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(graspPos - fingerCOM)
        reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
        reachDistz = np.abs(graspPos[-1] - fingerCOM[-1])
        zDist = np.abs(fingerCOM[-1] - self.init_fingerCOM[-1])

        placingDist = np.linalg.norm(objPos - placingGoal)

        def reachReward():
            reachRew = -reachDist# + min(actions[-1], -1)/50
            if reachDistxy < 0.04: #0.02
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zDist
            #incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50#max(actions[-1],0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            # if objPos[2] >= (heightTarget- tolerance):
            if objPos[2] >= (heightTarget- tolerance) and reachDist < 0.04:
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits
       
        def objGrasped(thresh = 0):
            sensorData = self.data.sensordata
            return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        def orig_pickReward():       
            # hScale = 50
            hScale = 100#100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
            elif (reachDist < 0.04) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            # if self.placeCompleted or (self.pickCompleted and objGrasped()):
            if self.placeCompleted:
                return hScale*heightTarget
            elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeRewardMove():
            # c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            placeRew = max(placeRew,0)
            if mode == 'general':
                cond = self.pickCompleted and objGrasped()
            else:
                cond = self.pickCompleted and (reachDist < 0.03) and not(objDropped())
            if cond:
                return [placeRew, placingDist]
            else:
                return [0 , placingDist]


        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()

        peg_pos = self.sim.model.body_pos[self.model.body_name2id('peg')]
        nut_pos = self.get_body_com('RoundNut')
        if abs(nut_pos[0] - peg_pos[0]) > 0.05 or \
                abs(nut_pos[1] - peg_pos[1]) > 0.05:
            placingDist = 0
            reachRew = 0
            reachDist = 0
            pickRew = heightTarget*100

        # placeRew , placingDist, placingDistFinal = placeRewardDrop()
        placeRew , placingDist = placeRewardMove()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        success = (abs(nut_pos[0] - peg_pos[0]) > 0.05 or abs(nut_pos[1] - peg_pos[1]) > 0.05) or placingDist < 0.02
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, float(success)] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
