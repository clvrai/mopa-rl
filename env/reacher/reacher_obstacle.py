from env.base import BaseEnv


class ReacherObstacleEnv(BaseEnv):
    """ Reacher with Obstacles environment. """

    def __init__(self):
        super().__init__("reacher_obstacle.xml")

    def _reset(self):
        self._set_camera_position(0, [0, -1.0, 1.0])
        self._set_camera_rotation(0, [0, 0, 0])
        return {}

    def _step(self, action):
        self._do_simulation(action)
        return {}, 0, False, {}
