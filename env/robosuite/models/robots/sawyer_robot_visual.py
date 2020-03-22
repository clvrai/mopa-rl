import numpy as np
from env.robosuite.models.robots.robot import Robot
from env.robosuite.utils.mjcf_utils import xml_path_completion, array_to_string
import copy


class SawyerVisual(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/sawyer/robot_visual.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}_visual".format(x) for x in range(7)]

    @property
    def bodies(self):
        return ["right_l{}_visual".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    def get_visual(self, name=None, site=False):
        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        return visual

