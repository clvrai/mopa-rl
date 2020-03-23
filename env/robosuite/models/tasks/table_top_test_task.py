from env.robosuite.models.tasks import Task, UniformRandomSampler, UniformRandomPegsSampler
from env.robosuite.utils.mjcf_utils import new_joint, array_to_string
from collections import OrderedDict


class TableTopTestTask(Task):
    """
    Creates MJCF model of a tabletop task.

    A tabletop task consists of one robot interacting with a variable number of
    objects placed on the tabletop. This class combines the robot, the table
    arena, and the objetcts into a single MJCF model.
    """

    def __init__(self, mujoco_arena, mujoco_robot, mujoco_objects, visual_objects, initializer=None):
        """
        Args:
            mujoco_arena: MJCF model of robot workspace
            mujoco_robot: MJCF model of robot model
            mujoco_objects: a list of MJCF models of physical objects
            initializer: placement sampler to initialize object positions.
        """
        super().__init__()

        self.merge_arena(mujoco_arena)
        self.merge_robot(mujoco_robot)
        self.merge_objects(mujoco_objects)
        self.merge_visual(OrderedDict(visual_objects))
        self.visual_objects = visual_objects
        if initializer is None:
            initializer = UniformRandomSampler()
        mjcfs = [x for _, x in self.mujoco_objects.items()]

        self.initializer = initializer
        self.initializer.setup(mjcfs, self.table_top_offset, self.table_size)

        self.visual_initializer = UniformRandomPegsSampler(
                x_range=[0.5, 1.],
                y_range=[0.5, 1.],
                z_range=[0.1, 0.2],
                ensure_object_boundary_in_range=False,
                z_rotation=False,
        )
        visual_mjcfs = {}
        for k, v in self.visual_objects:
            visual_mjcfs[k] = v
        self.visual_initializer.setup(visual_mjcfs, self.table_top_offset, self.table_size)

    def merge_robot(self, mujoco_robot):
        """Adds robot model to the MJCF model."""
        self.robot = mujoco_robot
        self.merge(mujoco_robot)

    def merge_arena(self, mujoco_arena):
        """Adds arena model to the MJCF model."""
        self.arena = mujoco_arena
        self.table_top_offset = mujoco_arena.table_top_abs
        self.table_size = mujoco_arena.table_full_size
        self.merge(mujoco_arena)

    def merge_objects(self, mujoco_objects):
        """Adds physical objects to the MJCF model."""
        self.mujoco_objects = mujoco_objects
        self.objects = []  # xml manifestation
        self.targets = []  # xml manifestation
        self.max_horizontal_radius = 0

        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_collision(name=obj_name, site=True)
            obj.append(new_joint(name=obj_name, type="free"))
            self.objects.append(obj)
            self.worldbody.append(obj)

            self.max_horizontal_radius = max(
                self.max_horizontal_radius, obj_mjcf.get_horizontal_radius()
            )

    def merge_visual(self, mujoco_objects):
        """Adds visual objects to the MJCF model."""
        self.visual_obj_mjcf = []
        for obj_name, obj_mjcf in mujoco_objects.items():
            self.merge_asset(obj_mjcf)
            # Load object
            obj = obj_mjcf.get_visual(name=obj_name, site=False)
            self.visual_obj_mjcf.append(obj)
            self.worldbody.append(obj)

    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        pos_arr, quat_arr = self.initializer.sample()
        for i in range(len(self.objects)):
            self.objects[i].set("pos", array_to_string(pos_arr[i]))
            self.objects[i].set("quat", array_to_string(quat_arr[i]))

    def place_visual(self):
        pos_arr, quat_arr = self.visual_initializer.sample()
        for i, (_, obj_mjcf) in enumerate(self.visual_objects):
            self.visual_obj_mjcf[i].set("pos", array_to_string(pos_arr[i]))
