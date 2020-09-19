import numpy as np

from util.gym import action_size
from util.logger import logger
from motion_planners.sampling_based_planner import SamplingBasedPlanner


class PlannerAgent:
    def __init__(
        self,
        config,
        ac_space,
        non_limited_idx=None,
        passive_joint_idx=[],
        ignored_contacts=[],
        planner_type=None,
        goal_bias=0.05,
        is_simplified=False,
        simplified_duration=0.1,
        range_=None,
    ):

        self._config = config
        self.planner = SamplingBasedPlanner(
            config,
            config._xml_path,
            action_size(ac_space),
            non_limited_idx,
            planner_type=planner_type,
            passive_joint_idx=passive_joint_idx,
            ignored_contacts=ignored_contacts,
            contact_threshold=config.contact_threshold,
            goal_bias=goal_bias,
            is_simplified=is_simplified,
            simplified_duration=simplified_duration,
            range_=range_,
        )

        self._is_simplified = is_simplified
        self._simplified_duration = simplified_duration

    def plan(self, start, goal, timelimit=None, attempts=15):
        config = self._config
        if timelimit is None:
            timelimit = config.timelimit
        traj, states, valid, exact = self.planner.plan(start, goal, timelimit)
        success = valid and exact

        if success:
            return traj[1:], success, valid, exact
        else:
            return traj, success, valid, exact

    def get_planner_status(self):
        return self.planner.get_planner_status()

    def isValidState(self, state):
        return self.planner.isValidState(state)
