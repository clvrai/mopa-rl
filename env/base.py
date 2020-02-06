import os
import time
import logging
import traceback
from collections import OrderedDict

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


class BaseEnv(gym.Env):
    """ Base class for MuJoCo environments. """

    def __init__(self, xml_path, **kwargs):
        """ Initializes class with configuration. """
        # default env config
        self._env_config = {
            "frame_skip": kwargs['frame_skip'],
            "ctrl_reward": 1e-3,
            "init_randomness": 1e-5,
            "max_episode_steps": kwargs['max_episode_steps'],
            "unstable_penalty": 0,
            "reward_type": kwargs['reward_type'],
            "distance_threshold": kwargs['distance_threshold'],
            "is_rgb": kwargs['is_rgb']
        }

        logger.setLevel(logging.INFO)

        self.render_mode = 'no' # ['no', 'human', 'rgb_array']
        self._screen_width = kwargs['screen_width']
        self._screen_height = kwargs['screen_height']
        self._seed = kwargs['seed']
        self._gym_disable_underscore_compat = True
        self._action_repeat = kwargs['action_repeat']
        self._img_height = kwargs['img_height']
        self._img_width = kwargs['img_width']

        # Load model
        self._load_model(xml_path)

    def _load_model(self, xml_path):
        if not xml_path.startswith('/'):
            xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'xml', xml_path)

        if not os.path.exists(xml_path):
            raise IOError('Model file ({}) does not exist'.format(xml_path))

        model = mujoco_py.load_model_from_path(xml_path)

        self._frame_skip = self._env_config["frame_skip"]
        self.sim = mujoco_py.MjSim(model, nsubsteps=self._frame_skip)
        self.model = self.sim.model
        self.data = self.sim.data
        self._viewer = None
        self.xml_path = xml_path

        # State
        logger.info('initial qpos: {}'.format(self.sim.data.qpos.ravel()))
        logger.info('initial qvel: {}'.format(self.sim.data.qvel.ravel()))

        # Action
        num_actions = self.sim.model.nu
        is_limited = self.sim.model.actuator_ctrllimited.ravel().astype(np.bool)
        control_range = self.sim.model.actuator_ctrlrange
        minimum = np.full(num_actions, fill_value=-np.inf, dtype=np.float)
        maximum = np.full(num_actions, fill_value=np.inf, dtype=np.float)
        minimum[is_limited], maximum[is_limited] = control_range[is_limited].T
        self._minimum = minimum
        self._maximum = maximum
        logger.info('is_limited: {}'.format(is_limited))
        logger.info('control_range: {}'.format(control_range[is_limited].T))
        self.action_space = spaces.Dict([
            ('default', spaces.Box(low=minimum, high=maximum, dtype=np.float32))
        ])

        self.joint_sapce = spaces.Dict([
            ('default', spaces.Box(low=-3, high=-3, shape=(self.model.nq,), dtype=np.float32))
        ])

        # Camera
        self._camera_name = 'cam0'
        self._obs_camera_name = 'cam1'
        self._camera_id = self.sim.model.camera_names.index(self._camera_name)
        self._obs_camera_id = self.sim.model.camera_names.index(self._obs_camera_name)

    @property
    def dt(self):
        return self.model.opt.timestep * self._frame_skip

    @property
    def max_episode_steps(self):
        return self._env_config["max_episode_steps"]

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_size(self):
        return self.sim.model.nu

    def reset(self):
        self.sim.reset()
        if self.render_mode == 'human':
            self._viewer = self._get_viewer()
            self._viewer_reset()
        ob = self._reset()
        self._after_reset()
        return ob

    def _init_random(self, size):
        r = self._env_config["init_randomness"]
        return np.random.uniform(low=-r, high=r, size=size)

    def _reset(self):
        raise NotImplementedError

    def _after_reset(self):
        self._episode_reward = 0
        self._episode_length = 0
        self._episode_time = time.time()

        self._terminal = False
        self._success = False
        self._fail = False

        #with self.model.disable('actuation'):
        #    self.forward()

    def step(self, action):
        self._before_step()
        if isinstance(action, list):
            action = {key: val for ac_i in action for key, val in ac_i.items()}
        if isinstance(action, OrderedDict):
            action = np.concatenate([action[key] for key in self.action_space.spaces.keys() if key in action])
        ob, reward, done, info = self._step(action)
        done, info, penalty = self._after_step(reward, done, info)
        return ob, reward + penalty, done, info

    def _before_step(self):
        pass

    def _step(self, action):
        raise NotImplementedError

    def _after_step(self, reward, terminal, info):
        step_log = dict(info)
        self._terminal = terminal
        penalty = 0

        if reward is not None:
            self._episode_reward += reward
            self._episode_length += 1

        if self._episode_length == self.max_episode_steps or self._fail:
            self._terminal = True
            if self._fail:
                self._fail = False
                penalty = -self._env_config["unstable_penalty"]

        if self._terminal:
            total_time = time.time() - self._episode_time
            step_log["episode_success"] = int(self._success)
            step_log["episode_reward"] = self._episode_reward + penalty
            step_log["episode_length"] = self._episode_length
            step_log["episode_time"] = total_time
            step_log["episode_unstable"] = penalty

        return self._terminal, step_log, penalty

    def _ctrl_reward(self, a):
        ctrl_reward = -self._env_config["ctrl_reward"] * np.square(a).sum()
        return ctrl_reward

    def _get_obs(self):
        raise NotImplementedError

    def set_env_config(self, env_config):
        self._env_config.update(env_config)

    def _render_callback(self):
        self.sim.forward()

    def _set_camera_position(self, cam_id, cam_pos):
        self.sim.model.cam_pos[cam_id] = cam_pos.copy()

    def _set_camera_rotation(self, cam_id, target_pos):
        cam_pos = self.sim.model.cam_pos[cam_id]
        forward = target_pos - cam_pos
        up = [forward[0], forward[1], (forward[0]**2 + forward[1]**2) / (-forward[2])]
        if forward[0] == 0 and forward[1] == 0:
            up = [0, 1, 0]
        q = T.lookat_to_quat(-forward, up)
        self.sim.model.cam_quat[cam_id] = T.convert_quat(q, to='wxyz')

    def render(self, mode='human', close=False):
        self._render_callback() # sim.forward()

        if mode == 'rgb_array':
            camera_obs = self.sim.render(camera_name=self._camera_name,
                                         width=self._screen_width,
                                         height=self._screen_height,
                                         depth=False)
            camera_obs = camera_obs[::-1, :, :] / 255.0
            assert np.sum(camera_obs) > 0, 'rendering image is blank'
            return camera_obs
        elif mode == 'human':
            self._get_viewer().render()
            return None
        return None

    def _viewer_reset(self):
        pass

    def _get_current_error(self, current_state, desired_state):
        return desired_state - current_state

    def _get_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self.sim)
            self._viewer.cam.fixedcamid = self._camera_id
            self._viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED
            self._viewer_reset()
        return self._viewer

    def close(self):
        if self._viewer is not None:
            self._viewer = None

    def _do_simulation(self, a):
        try:
            self.data.ctrl[:] = a
            for _ in range(self._frame_skip):
                self.sim.forward()
                self.sim.step()
        except Exception as e:
            logger.warn('[!] Warning: Simulation is unstable. The episode is terminated.')
            logger.warn(e)
            logger.warn(traceback.format_exc())
            self.reset()
            self._fail = True

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        self.sim.step()

    def _get_pos(self, name):
        if name in self.model.body_names:
            return self.data.get_body_xpos(name).copy()
        if name in self.model.geom_names:
            return self.data.get_geom_xpos(name).copy()
        raise ValueError

    def _set_pos(self, name, pos):
        if name in self.model.body_names:
            body_idx = self.model.body_name2id(name)
            self.model.body_pos[body_idx] = pos[:]
            return
        if name in self.model.geom_names:
            geom_idx = self.model.geom_name2id(name)
            self.model.geom_pos[geom_idx][0:3] = pos[:]
            return
        raise ValueError

    def _get_quat(self, name):
        if name in self.model.body_names:
            return self.data.get_body_xquat(name).copy()
        raise ValueError

    def _get_right_vector(self, name):
        if name in self.model.geom_names:
            return self.data.get_geom_xmat(name)[0].copy()
        raise ValueError

    def _get_forward_vector(self, name):
        if name in self.model.geom_names:
            return self.data.get_geom_xmat(name)[1].copy()
        raise ValueError

    def _get_up_vector(self, name):
        if name in self.model.geom_names:
            return self.data.get_geom_xmat(name)[2].copy()
        raise ValueError

    def _set_quat(self, name, quat):
        if name in self.model.body_names:
            body_idx = self.model.body_name2id(name)
            self.model.body_quat[body_idx] = quat[:]
            return
        if name in self.model.geom_names:
            geom_idx = self.model.geom_name2id(name)
            self.model.geom_quat[geom_idx][0:4] = quat[:]
            return
        raise ValueError

    def _get_distance(self, name1, name2):
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_size(self, name):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.model.geom_size[geom_idx, :].copy()

    def _set_size(self, name, size):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_size[geom_idx, :] = size

    def _get_geom_type(self, name):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                return self.model.geom_type[geom_idx].copy()

    def _set_geom_type(self, name, geom_type):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_type[geom_idx] = geom_type

    def _get_qpos(self, name):
        object_qpos = self.data.get_joint_qpos(name)
        return object_qpos.copy()

    def _set_qpos(self, name, pos, rot=[1, 0, 0, 0]):
        object_qpos = self.data.get_joint_qpos(name)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = pos
        object_qpos[3:] = rot
        self.data.set_joint_qpos(name, object_qpos)

    def _set_color(self, name, color):
        body_idx1 = self.model.body_name2id(name)
        for geom_idx, body_idx2 in enumerate(self.model.geom_bodyid):
            if body_idx1 == body_idx2:
                self.model.geom_rgba[geom_idx, 0:len(color)] = color

    def _mass_center(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xpos = self.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))

    def on_collision(self, ref_name, geom_name=None):
        mjcontacts = self.data.contact
        ncon = self.data.ncon
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 = self.model.geom_id2name(ct.geom1)
            g2 = self.model.geom_id2name(ct.geom2)
            if g1 is None or g2 is None:
                continue # geom_name can be None
            if geom_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and \
                    (g1.find(geom_name) >= 0 or g2.find(geom_name) >= 0):
                    return True
            else:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0):
                    return True
        return False

