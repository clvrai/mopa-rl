from env.base import BaseEnv
import numpy as np


class ReacherObstacleEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self):
        super().__init__("reacher_obstacle.xml")
        self.obstacle_names = ["obstacle1", "obstacle2", "obstacle3",
                               "obstacle4", "obstacle5", "obstacle6",
                               "obstacle7", "obstacle8"]

    def _reset(self):
        self._set_camera_position(0, [0, -1.0, 1.0])
        self._set_camera_rotation(0, [0, 0, 0])
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.sim.data.qpos.ravel()
        while True:
            self.goal = np.random.uniform(low=-.5, high=.5, size=2)
            # not too close and far from the root
            if np.linalg.norm(self.goal) < 0.4 and np.linalg.norm(self.goal) > 0.2:
                break
        qpos[-2:] = self.goal
        qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv) + self.sim.data.qvel.ravel()
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self._reset_obstacles()
        return self._get_obs()

    def _reset_obstacles(self):
        for name in self.obstacle_names:
            while True:
                pos = np.random.uniform(-0.3, 0.3, size=2)
                # not too close and far from the root, and not overlapped with a target
                if np.linalg.norm(pos) < 0.25 and np.linalg.norm(pos) > 0.05 \
                        and np.linalg.norm(pos-self.goal) > 0.05:
                    break
            self._set_pos(name, np.concatenate([pos, np.array([0.01])]))
            self._set_size(name, np.concatenate([np.random.uniform(low=0.015, high=0.035, size=2), np.array([0.05])]))

    def _get_obstacle_states(self):
        obstacle_states = []
        for name in self.obstacle_names:
            obstacle_states.extend(self._get_pos(name)[:2])
        return np.array(obstacle_states)


    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self._get_obstacle_states(),
            self._get_pos("fingertip") - self._get_pos("target")
        ])

    def _step(self, action):
        reward_dist = -self._get_distance("fingertip", "target")
        reward_ctrl = self._ctrl_reward(action)
        reward = reward_dist + reward_ctrl
        self._do_simulation(action)
        obs = self._get_obs()
        done = False
        return obs, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
