import os, sys
import numpy as np

from env.base import BaseEnv
import env.robosuite.utils.transform_utils as T

from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml
from env.robosuite.models.grippers import gripper_factory
from env.robosuite.models.robots import Sawyer, SawyerIndicator
from env.robosuite.utils.mjcf_utils import new_joint, array_to_string

from collections import OrderedDict

import gym
from gym import spaces, error


class SawyerEnv(BaseEnv):
    """Initializes a Sawyer robot environment."""

    def __init__(
        self,
        gripper_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        control_freq=10,
        max_episode_steps=1000,
        ignore_done=False,
        camera_name="frontview",
        img_height=256,
        img_width=256,
        img_depth=False,
        use_robot_indicator=False,
        **kwargs):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.
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
            use_camera_obs (bool): if True, every observation includes a
                rendered image.
            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.
            camera_height (int): height of camera frame.
            camera_width (int): width of camera frame.
            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        self.control_freq = control_freq
        self.use_robot_indicator = use_robot_indicator

        xml_name = kwargs['env'].replace("-v0", "")
        xml_name = xml_name.replace("-", "_")
        xml_path = os.path.join(os.path.dirname(__file__), '../assets', 'xml', xml_name+'.xml')
        self.xml_path = xml_path
        self._load_model()
        self.model.save_model(self.xml_path)


        super().__init__(
            self.xml_path,
            control_freq=control_freq,
            max_episode_steps=max_episode_steps,
            ignore_done=ignore_done,
            camera_name=camera_name,
            img_height=img_height,
            img_width=img_width,
            img_depth=img_depth,
            **kwargs)


        # Action
        num_actions = self.dof
        is_limited = np.array([True] * self.dof)
        minimum = np.ones(self.dof) * -1.
        maximum = np.ones(self.dof) * 1.

        self._minimum = minimum
        self._maximum = maximum
        self.action_space = spaces.Dict([
            ('default', spaces.Box(low=minimum, high=maximum, dtype=np.float32))
        ])

        jnt_range = self.sim.model.jnt_range[:num_actions]
        is_jnt_limited = self.sim.model.jnt_limited[:num_actions].astype(np.bool)
        jnt_minimum = np.full(num_actions, fill_value=-np.inf, dtype=np.float)
        jnt_maximum = np.full(num_actions, fill_value=np.inf, dtype=np.float)
        jnt_minimum[is_jnt_limited], jnt_maximum[is_jnt_limited] = jnt_range[is_jnt_limited].T
        jnt_minimum[np.invert(is_jnt_limited)] = -3.14
        jnt_maximum[np.invert(is_jnt_limited)] = 3.14
        self._is_jnt_limited = is_jnt_limited

        end_time = self.cur_time + self.control_timestep
        self._frame_skip = int(end_time / self.model_timestep)

        self._prev_state = None
        self._i_term = None

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Sawyer()
        if self.use_robot_indicator:
            self.mujoco_robot_indicator = SawyerIndicator()
        else:
            self.mujoco_robot_indicator = None

        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        self._load_model()
        super()._reset_internal()
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = MjSim(self.mjpy_model)
        self.data = self.sim.data
        self.initialize_time(self.control_freq)

        self._get_reference()

        self.sim.data.qpos[self.ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

        if self.has_gripper:
            self.sim.data.qpos[
                self.ref_gripper_joint_pos_indexes
            ] = self.gripper.init_qpos


    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        if self.use_robot_indicator:
            self.robot_indicator_joints = list(self.mujoco_robot_indicator.joints)
            self.ref_indicator_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.robot_indicator_joints
            ]
            self.ref_indicator_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.robot_indicator_joints
            ]

        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self.ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self.ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")
            ]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index : index + 3] = pos

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

        # clip actions into valid range
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
        # if self.use_indicator_object:
        #     self.sim.data.qfrc_applied[
        #         self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #     ] = self.sim.data.qfrc_bias[
        #         self._ref_indicator_vel_low : self._ref_indicator_vel_high
        #     ]

    def _get_control(self, state, prev_state, target_vel):
        alpha = 0.95

        p_term = self._kp * (state - self.sim.data.qpos[self.ref_joint_pos_indexes])
        d_term = self._kd * (target_vel * 0 - self.sim.data.qvel[self.ref_joint_pos_indexes])
        self._i_term = alpha * self._i_term + self._ki * (prev_state - self.sim.data.qpos[self.ref_joint_pos_indexes])
        action = p_term + d_term + self._i_term

        return action

    def _step(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        assert len(action) == self.dof, "environment got invalid action dimension"

        if self._prev_state is None:
            self._prev_state = self.sim.data.qpos[self.ref_joint_pos_indexes]

        if self._i_term is None:
            self._i_term = np.zeros_like(self.mujoco_robot.dof)

        n_inner_loop = int(self.dt / self.sim.model.opt.timestep)

        desired_state = self._prev_state + action[:self.mujoco_robot.dof]
        target_vel = action[:self.mujoco_robot.dof] / self.dt

        for t in range(n_inner_loop):
            # gravity compensation
            self.sim.data.qfrc_applied[self.ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self.ref_joint_vel_indexes]

            self.sim.data.qfrc_applied[
                self._ref_target_vel_low : self._ref_target_vel_high+1
            ] = self.sim.data.qfrc_bias[
                self._ref_target_vel_low : self._ref_target_vel_high+1
            ]

            if self.use_indicator_object:
                self.sim.data.qfrc_applied[
                    self._ref_indicator_vel_low : self._ref_indicator_vel_high
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low : self._ref_indicator_vel_high
                ]

            arm_action = self._get_control(desired_state, self._prev_state, target_vel)
            gripper_action = self.gripper.format_action(np.array([action[-1]]))

            action = np.concatenate([arm_action, -gripper_action])
            self._do_simulation(action)

        # 1) RL policy
        # self._prev_state = None
        # 2) Planner policy
        self._prev_state = np.copy(desired_state)

        reward = self.reward(action)
        # done if number of elapsed timesteps is greater than horizon
        self._gripper_visualization()

        return self._get_obs(), reward, self._terminal, {}

    def _step(self, action):
        """
        (Optional) does gripper visualization after actions.
        """
        self._do_simulation()
        reward = self.reward(action)
        # done if number of elapsed timesteps is greater than horizon
        self._gripper_visualization()
        return self._get_obs(), reward, self._terminal, {}

    def _get_obs(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            robot-state: contains robot-centric information.
        """

        di = super()._get_obs()
        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self.ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self.ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self.ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self.ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            # add in gripper information
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])

        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self.ref_joint_pos_indexes] = jpos
        self.sim.forward()

    def set_robot_indicator_joint_positions(self, jpos):
        assert self.use_robot_indicator == True, "use_robot_indicator must be True."
        self.sim.data.qpos[self.ref_indicator_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name("right_l6")

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self.ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self.ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    def _robot_jpos_getter(self):
        """
        Helper function to pass to the ik controller for access to the
        current robot joint positions.
        """
        return np.array(self._joint_positions)


    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self.ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self.ref_joint_vel_indexes]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False

