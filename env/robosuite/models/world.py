from env.robosuite.models.base import MujocoXML


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__("./env/assets/robosuite/base.xml")
