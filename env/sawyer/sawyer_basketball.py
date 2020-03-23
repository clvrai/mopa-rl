from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from env.multitask_env import MultitaskEnv
from env.sawyer.base import SawyerXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

class SawyerBasketballEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=False,
            obs_type='plain',
            goal_low=(-0.1, 0.85, 0.15),
            goal_high=(0.1, 0.9+1e-7, 0.15),
            liftThresh = 0.3,
            rotMode='fixed',
            rewMode = 'orig',
            **kwargs
    ):

        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(-0.1, 0.6, 0.03)
        obj_high=(0.1, 0.7, 0.03)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.03], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        self.random_init = random_init
        self.max_path_length = 150
        self.rotMode = rotMode
        self.rewMode = rewMode
        self.liftThresh = liftThresh

        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
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
        return get_asset_full_path('sawyer_xyz/sawyer_basketball.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pickRew, placingDist = self.compute_reward(action, obs_dict, mode=self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        info = {'reachDist': reachDist, 'goalDist': placingDist, 'epRew' : reward, 'pickRew':pickRew, 'success': float(placingDist <= 0.08)}
        info['goal'] = self.goal
        return ob, reward, done, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
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
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('objGeom')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    




    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def adjust_initObjPos(self, orig_init_pos):
        #This is to account for meshes for the geom and object are not aligned
        #If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]


    def reset_model(self):
        self._reset_hand()
        
        basket_pos = self.goal.copy()
        self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = basket_pos
        self._state_goal = self.data.site_xpos[self.model.site_name2id('goal')]

        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            basket_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - basket_pos[:2]) < 0.15:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
                basket_pos = goal_pos[3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = basket_pos
            self._state_goal = self.data.site_xpos[self.model.site_name2id('goal')]
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.curr_path_length = 0
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
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

    def compute_reward(self, actions, obs, mode = 'orig'):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        goal = self._state_goal

        reachDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - goal)
        assert np.all(goal == self.get_site_pos('goal'))

        def reachReward():
            reachRew = -reachDist# + min(actions[-1], -1)/50
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.05: #0.02
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew
            #incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget- tolerance):
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
            hScale = 100
            # hScale = 1000
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and objGrasped():
                return hScale*heightTarget
            elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if mode == 'general':
                cond = self.pickCompleted and objGrasped()
            else:
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()
        placeRew , placingDist = placeReward()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        return [reward, reachDist, pickRew, placingDist]


    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
