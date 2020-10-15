import re
from collections import OrderedDict

import numpy as np
from gym import spaces

from env.base import BaseEnv


class PusherObstacleEnv(BaseEnv):
    """ Pusher with Obstacles environment. """

    def __init__(self, **kwargs):
        super().__init__("pusher_obstacle.xml", **kwargs)
        self.obstacle_names = list(
            filter(lambda x: re.search(r"obstacle", x), self.sim.model.body_names)
        )
        self._env_config.update({"success_reward": kwargs["success_reward"]})
        self.joint_names = ["joint0", "joint1", "joint2", "joint3"]
        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.joint_names
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.joint_names
        ]
        self.ref_indicator_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x + "-goal") for x in self.joint_names
        ]
        self.ref_dummy_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x + "-dummy") for x in self.joint_names
        ]

        self._ac_scale = 0.1
        self.max_world_size = [0.41, 0.41]
        self.min_world_size = [-0.41, -0.41]
        self._agent_colors = [
            self._get_color(body_name) for body_name in self.body_names
        ]

    def _reset(self):
        self._set_camera_position(0, [0, -0.7, 1.5])
        self._set_camera_rotation(0, [0, 0, 0])
        while True:
            goal = self.np_random.uniform(low=[-0.35, 0.13], high=[-0.24, 0.2], size=2)
            box = self.np_random.uniform(low=[-0.35, 0.13], high=[-0.24, 0.2], size=2)
            qpos = (
                self.np_random.uniform(low=-0.02, high=0.02, size=self.sim.model.nq)
                + self.sim.data.qpos.ravel()
            )
            qpos[-4:-2] = goal
            qpos[-2:] = box
            qvel = (
                self.np_random.uniform(low=-0.005, high=0.005, size=self.sim.model.nv)
                + self.sim.data.qvel.ravel()
            )
            qvel[-4:-2] = 0
            qvel[-2:] = 0
            self.set_state(qpos, qvel)
            if (
                self.sim.data.ncon == 0
                and self._get_distance("box", "target") > 0.1
                and goal[0] <= box[0]
            ):
                self.goal = goal
                self.box = box
                break
        return self._get_obs()

    @property
    def robot_joints(self):
        return ["joint0", "joint1", "joint2", "joint3"]

    @property
    def manipulation_geom(self):
        return ["box"]

    @property
    def robot_joints(self):
        return ["joint0", "joint1", "joint2", "joint3"]

    def visualize_goal_indicator(self, qpos):
        self.sim.data.qpos[self.ref_indicator_joint_pos_indexes] = qpos
        for body_name in self.body_names:
            key = body_name + "-goal"
            color = self._get_color(key)
            color[-1] = 0.3
            self._set_color(key, color)

    def visualize_dummy_indicator(self, qpos):
        self.sim.data.qpos[self.ref_dummy_joint_pos_indexes] = qpos
        for body_name in self.body_names:
            key = body_name + "-dummy"
            color = self._get_color(key)
            color[-1] = 0.3
            self._set_color(key, color)

    def reset_visualized_indicator(self):
        for body_name in self.body_names:
            for postfix in ["-goal", "-dummy"]:
                key = body_name + postfix
                color = self._get_color(key)
                color[-1] = 0.0
                self._set_color(key, color)

    def color_agent(self):
        for body_name in self.body_names:
            color = self._get_color(body_name)
            color = np.array([0.1, 0.3, 0.7, 1.0])
            self._set_color(body_name, color)

    def reset_color_agent(self):
        for i, body_name in enumerate(self.body_names):
            color = self._agent_colors[i]
            self._set_color(body_name, color)

    @property
    def body_names(self):
        return ["body0", "body1", "body2", "body3", "fingertip"]

    @property
    def manipulation_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.manipulation_geom]

    @property
    def body_geoms(self):
        return [
            "root",
            "link0",
            "link1",
            "link2",
            "link3",
            "fingertip0",
            "fingertip1",
            "fingertip2",
        ]

    @property
    def static_geoms(self):
        return [
            "obstacle1_geom",
            "obstacle2_geom",
            "obstacle3_geom",
            "obstacle4_geom",
            "obstacle5_geom",
            "obstacle6_geom",
            "obstacle7_geom",
        ]

    @property
    def static_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.static_geoms]

    @property
    def agent_geoms(self):
        return self.body_geoms

    @property
    def agent_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.agent_geoms]

    @property
    def static_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.static_geoms]

    def initialize_joints(self):
        while True:
            qpos = (
                self.np_random.uniform(low=-0.1, high=0.1, size=self.sim.model.nq)
                + self.sim.data.qpos.ravel()
            )
            qpos[-4:-2] = self.goal
            qpos[-2:] = self.box
            self.set_state(qpos, self.sim.data.qvel.ravel())
            if self.sim.data.ncon == 0:
                break

    def _get_obstacle_states(self):
        obstacle_states = []
        obstacle_size = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
            obstacle_size.extend(self._get_size(name)[:2])
        return np.concatenate([obstacle_states, obstacle_size])

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[self.ref_joint_pos_indexes]
        return OrderedDict(
            [
                (
                    "default",
                    np.concatenate(
                        [
                            np.cos(theta),
                            np.sin(theta),
                            self.sim.data.qpos.flat[-2:],  # box qpos
                            self.sim.data.qvel.flat[self.ref_joint_vel_indexes],
                            self.sim.data.qvel.flat[-2:],  # box vel
                        ]
                    ),
                ),
                ("fingertip", self._get_pos("fingertip")[:-1]),
                ("goal", self.sim.data.qpos.flat[-4:-2]),
            ]
        )

    @property
    def observation_space(self):
        return spaces.Dict(
            [
                ("default", spaces.Box(shape=(16,), low=-1, high=1, dtype=np.float32)),
                ("fingertip", spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)),
                ("goal", spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)),
            ]
        )

    @property
    def get_joint_positions(self):
        """
        The joint position except for goal states
        """
        return self.sim.data.qpos.ravel()[self.ref_joint_pos_indexes]

    def compute_reward(self, action):
        info = {}
        reward_type = self._env_config["reward_type"]
        reward_reach = 0.0
        reward_push = 0.0
        dist_box_to_gripper = np.linalg.norm(
            self._get_pos("box") - self.sim.data.get_site_xpos("fingertip")
        )
        if dist_box_to_gripper < 0.1:
            reward_reach += 0.1 * (1 - np.tanh(5 * dist_box_to_gripper))
        if self._get_distance("box", "target") < 0.1:
            reward_push += 0.3 * (1 - np.tanh(5 * self._get_distance("box", "target")))
        reward = reward_reach + reward_push
        info = dict(reward_reach=reward_reach, reward_push=reward_push)

        if self._get_distance("box", "target") < self._env_config["distance_threshold"]:
            self._success = True
            self._terminal = True
            reward += self._env_config["success_reward"]

        return reward, info

    def _step(self, action, is_planner=False):
        """
        Args:
            action (numpy array): The array should have the corresponding elements.
                0-6: The desired change in joint state (radian)
        """

        info = {}
        done = False

        if not is_planner or self._prev_state is None:
            self._prev_state = self.get_joint_positions

        if is_planner:
            rescaled_ac = np.clip(action, -self._ac_scale, self._ac_scale)
        else:
            rescaled_ac = np.clip(
                action * self._ac_scale, -self._ac_scale, self._ac_scale
            )

        if not is_planner:
            desired_state = (
                self._prev_state + self._ac_scale * action
            )  # except for gripper action
        else:
            desired_state = self._prev_state + action

        desired_state = self._prev_state + action  # except for gripper action

        n_inner_loop = int(self._frame_dt / self.dt)

        target_vel = (desired_state - self._prev_state) / self._frame_dt
        for t in range(n_inner_loop):
            ac = self._get_control(desired_state, self._prev_state, target_vel)
            self._do_simulation(ac)

        self._prev_state = np.copy(desired_state)
        reward, info = self.compute_reward(action)

        return self._get_obs(), reward, self._terminal, info
