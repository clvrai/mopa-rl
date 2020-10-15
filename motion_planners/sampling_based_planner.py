import os, sys

import numpy as np
import subprocess
from threading import Lock, Thread
import yaml
from motion_planners.planner import PyKinematicPlanner
from util.env import joint_convert


class SamplingBasedPlanner:
    def __init__(
        self,
        config,
        xml_path,
        num_actions,
        non_limited_idx,
        planner_type=None,
        passive_joint_idx=[],
        glue_bodies=[],
        ignored_contacts=[],
        contact_threshold=0.0,
        goal_bias=0.05,
        is_simplified=False,
        simplified_duration=0.1,
        range_=None,
    ):
        self.config = config
        if planner_type is None:
            planner_type = config.planner_type
        if range_ is None:
            range_ = config.range
        self.planner = PyKinematicPlanner(
            xml_path.encode("utf-8"),
            planner_type.encode("utf-8"),
            num_actions,
            config.planner_objective.encode("utf-8"),
            config.threshold,
            range_,
            passive_joint_idx,
            glue_bodies,
            ignored_contacts,
            contact_threshold,
            goal_bias,
            is_simplified,
            simplified_duration,
            config.seed,
        )
        self.non_limited_idx = non_limited_idx

    def convert_nonlimited(self, state):
        if self.non_limited_idx is not None:
            for idx in self.non_limited_idx:
                state[idx] = joint_convert(state[idx])
        return state

    def isValidState(self, state):
        return self.planner.isValidState(state)

    def plan(self, start, goal, timelimit=1.0):
        valid_state = True
        exact = True
        converted_start = self.convert_nonlimited(start.copy())
        converted_goal = self.convert_nonlimited(goal.copy())
        states = np.array(self.planner.plan(converted_start, converted_goal, timelimit))

        if np.unique(states).size == 1:
            if states[0][0] == -5:
                valid_state = False
            if states[0][0] == -4:
                exact = False
            return states, states, valid_state, exact

        traj = [start]
        pre_state = states[0]
        for _, state in enumerate(states[1:]):
            # converted_pre_state = self.convert_nonlimited(pre_state.copy())
            tmp_state = traj[-1] + (state - pre_state)
            if self.non_limited_idx is not None:
                for idx in self.non_limited_idx:
                    if abs(state[idx] - pre_state[idx]) > 3.14:
                        if pre_state[idx] > 0 and state[idx] <= 0:
                            # if traj[-1][idx] < 0:
                            tmp_state[idx] = traj[-1][idx] + (
                                3.14 - pre_state[idx] + state[idx] + 3.14
                            )
                            # else:
                            #     tmp_state[idx] = traj[-1][idx] - (3.14-pre_state[idx] + state[idx] + 3.14)
                            # tmp_state[idx] = traj[-1][idx] + 3.14 + state[idx]
                        elif pre_state[idx] < 0 and state[idx] > 0:
                            # if traj[-1][idx] < 0:
                            tmp_state[idx] = traj[-1][idx] - (
                                3.14 - state[idx] + pre_state[idx] + 3.14
                            )
                            # else:
                            #     tmp_state[idx] = traj[-1][idx] + (3.14-state[idx] + pre_state[idx] + 3.14)

                            # tmp_state[idx] = traj[-1][idx] - 3.14 + state[idx]
            pre_state = state
            traj.append(tmp_state)
        return np.array(traj), states, valid_state, exact

    def remove_collision(self, geom_id, contype, conaffinity):
        self.planner.removeCollision(geom_id, contype, conaffinity)

    def get_planner_status(self):
        return self.planner.getPlannerStatus().decode("utf-8")
