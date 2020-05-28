from env.robosuite.models.world import MujocoWorldBase
from env.robosuite.utils.mjcf_utils import new_joint, array_to_string, new_geom, new_body, new_site
import numpy as np

class Task(MujocoWorldBase):
    """
    Base class for creating MJCF model of a task.

    A task typically involves a robot interacting with objects in an arena
    (workshpace). The purpose of a task class is to generate a MJCF model
    of the task by combining the MJCF models of each component together and
    place them to the right positions. Object placement can be done by
    ad-hoc methods or placement samplers.
    """

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        pass

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        pass

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        pass

    def merge_visual(self, mujoco_objects):
        """Adds visual objects to the MJCF model."""

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pass

    def place_visual(self):
        """Places visual objects randomly until no collisions or max iterations hit."""
        pass

    def add_target(self):
        body = new_body(name='target', pos=np.array([0.6, 0, 1.2]))
        body.append(new_joint(type='slide', axis='1 0 0',  name='target_x', pos='0 0 0', limited='true', range='-1 1', ref='0.6'))
        body.append(new_joint(type='slide', axis='0 1 0',  name='target_y', pos='0 0 0', limited='true', range='-1 1', ref='0'))
        body.append(new_joint(type='slide', axis='0 0 1',  name='target_z', pos='0 0 0', limited='true', range='-1 1.5', ref='1.2'))
        body.append(
            new_geom(
                'sphere',
                [0.02],
                rgba=[0, 1, 0, 0.5],
                group=1,
                contype="0",
                conaffinity="0",
                pos=np.array([0, 0, 0]),
                name='target_geom'
            )
        )
        self.worldbody.append(body)
