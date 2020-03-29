from collections import OrderedDict
import numpy as np

from env.robosuite.utils.transform_utils import convert_quat
from env.robosuite.sawyer import SawyerEnv

from env.robosuite.models.arenas.table_arena import TableArena
from env.robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
from env.robosuite.models.robots import Sawyer, SawyerIndicator
from env.robosuite.models.tasks import TableTopTargetTask, UniformRandomSampler

class TargetVisualObject(MujocoXMLObject):
    def __init__(self):
        super().__init__("./env/assets/xml/common/target.xml")

class SawyerTestEnv(SawyerEnv):
    """
    This class corresponds to the stacking task for the Sawyer robot arm.
    """

    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.82),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
        gripper_visualization=True,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        use_robot_indicator=True,
        **kwargs):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.
            table_full_size (3-tuple): x, y, and z dimensions of the table.
            table_friction (3-tuple): the three mujoco friction parameters for
                the table.
            use_camera_obs (bool): if True, every observation includes a
                rendered image.
            use_object_obs (bool): if True, include object (cube) information in
                the observation.
            reward_shaping (bool): if True, use dense rewards.
            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.
            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.
            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.
            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.
            has_offscreen_renderer (bool): True if using off-screen rendering.
            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.
            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.
            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.
            horizon (int): Every episode lasts for exactly @horizon timesteps.
            ignore_done (bool): True if never terminating the environment (ignore @horizon).
            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.
            camera_height (int): height of camera frame.
            camera_width (int): width of camera frame.
            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                ensure_object_boundary_in_range=False,
                z_rotation=False,
            )

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
            use_robot_indicator=use_robot_indicator,
            **kwargs
        )

        # reward configuration
        self.reward_shaping = reward_shaping

        # information of objects
        # self.object_names = [o['object_name'] for o in self.object_metadata]
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [
            self.sim.model.site_name2id(ob_name) for ob_name in self.object_names
        ]

        # id of grippers for contact checking
        self.finger_names = self.gripper.contact_geoms()

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys()
        self.collision_check_geom_ids = [
            self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names
        ]

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])
        if self.use_robot_indicator:
            self.mujoco_robot_indicator.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()


        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])


        # initialize objects of interest
        #target = TargetObject()
        self.mujoco_objects = OrderedDict()

        # task includes arena, robot, and objects of interest
        self.model = TableTopTargetTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
            mujoco_robot_indicator=self.mujoco_robot_indicator
        )

        self.model.place_objects()
        # self.add_visual_sawyer()

    def _pre_action(self, action):
        pass
        """
        Overrides the superclass method to actuate the robot with the 
        passed joint velocities and gripper control.
        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired 
                normalized joint velocities and if the robot has 
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        # # clip actions into valid range
        # assert len(action) == self.dof, "environment got invalid action dimension"
        # low, high = self.action_spec
        # action = np.clip(action, low, high)
        #
        # if self.has_gripper:
        #     arm_action = action[: self.mujoco_robot.dof]
        #     gripper_action_in = action[
        #         self.mujoco_robot.dof : self.mujoco_robot.dof + self.gripper.dof
        #     ]
        #     gripper_action_actual = self.gripper.format_action(gripper_action_in)
        #     action = np.concatenate([arm_action, gripper_action_actual])
        #
        # # rescale normalized action to control ranges
        # ctrl_range = self.sim.model.actuator_ctrlrange
        # bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        # weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        # applied_action = bias + weight * action
        # self.sim.data.ctrl[:] = applied_action
        #
        # # gravity compensation
        # self.sim.data.qfrc_applied[
        #     self.ref_joint_vel_indexes
        # ] = self.sim.data.qfrc_bias[self.ref_joint_vel_indexes]
        #
        # self.sim.data.qfrc_applied[
        #     self._ref_target_vel_low : self._ref_target_vel_high+1
        # ] = self.sim.data.qfrc_bias[
        #     self._ref_target_vel_low : self._ref_target_vel_high+1
        # ]
        #
        # if self.use_indicator_object:
        #     self.sim.data.qfrc_applied[
        #         self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #     ] = self.sim.data.qfrc_bias[
        #         self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #     ]

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        ind_qpos = (self.sim.model.get_joint_qpos_addr("target_x"), self.sim.model.get_joint_qpos_addr('target_z'))
        self._ref_target_pos_low, self._ref_target_pos_high = ind_qpos

        ind_qvel = (self.sim.model.get_joint_qvel_addr("target_x"), self.sim.model.get_joint_qvel_addr('target_z'))
        self._ref_target_vel_low, self._ref_target_vel_high = ind_qvel

        self.target_id = self.sim.model.body_name2id("target")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()
        # self.model.place_visual()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = np.array(init_pos)

    def initialize_joints(self):
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self.ref_joint_pos_indexes] = np.array(init_pos)


    def reward(self, action):
        """
        Reward function for the task.
        The dense reward has five components.
            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube
            Aligning: in [0, 0.5], encourages aligning one cube over the other
            Stacking: in {0, 2}, non-zero if cube is stacked on other cube
        The sparse reward only consists of the stacking component.
        However, the sparse reward is either 0 or 1.
        Args:
            action (np array): unused for this task
        Returns:
            reward (float): the reward
        """

        return 0

    def _get_obs(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_obs()
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # low-level object information
        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in self.finger_names
                or self.sim.model.geom_id2name(contact.geom2) in self.finger_names
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to nearest object
        if self.gripper_visualization:
            rgba = np.zeros(4)
            rgba[0] = 1
            rgba[1] = 0
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
