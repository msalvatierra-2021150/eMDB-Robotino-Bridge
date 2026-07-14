"""Navigation planning: evaluate a tag approach pose via ComputePathToPose.

Plans the observation-side approach to a remembered tag and hands the
selected candidate off to navigation_execution once it is confirmed
reachable.
"""

import copy
import math
from typing import Any, Dict, List

from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose

from robotino_emdb_interfaces.msg import RobotinoPolicyOutcome, RobotinoSelectedPolicy

from . import constants

MIN_DISTANCE_EPSILON_M = 0.001


class NavigationPlanningMixin:
    """Requires from the host class:

    Attributes: execution, execution_generation, candidate_plan_candidates,
        candidate_plan_index, compute_path_client, enable_nav2_execution,
        active_path_goal_handle.
    Methods: set_exploration_enabled(), publish_simple_outcome(),
        resume_after_failed_execution_if_configured(), goal_interval_ok(),
        get_robot_position_from_tf(), get_approach_axis(),
        make_pose_stamped(), get_current_energy(), get_current_reward(),
        get_execution(), finish_execution(), compute_path_length(),
        send_navigation_goal().
    Constants: STAGE_PLANNING.
    """

    def start_tag_navigation(
        self,
        policy: RobotinoSelectedPolicy,
        approach_standoff_distance: float,
    ) -> None:
        """Plan both sides of a remembered tag and execute the best path."""
        self.set_exploration_enabled(False)

        if not bool(policy.use_nav2):
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_DOES_NOT_USE_NAV2,
            )
            self.resume_after_failed_execution_if_configured()
            return

        if not self.enable_nav2_execution:
            self.get_logger().warn(
                "Nav2 execution is disabled. Set "
                "enable_nav2_execution:=true to send goals."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_EXECUTION_DISABLED,
            )
            self.resume_after_failed_execution_if_configured()
            return

        if not self.goal_interval_ok():
            self.get_logger().debug(
                "Policy ignored because minimum Nav2 goal interval has not elapsed."
            )
            self.resume_after_failed_execution_if_configured()
            return

        target_x = float(policy.target_x_map)
        target_y = float(policy.target_y_map)
        observation_x = float(policy.last_seen_robot_x_map)
        observation_y = float(policy.last_seen_robot_y_map)

        robot_position = self.get_robot_position_from_tf()
        if robot_position is None:
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_NO_ROBOT_POSE,
            )
            self.resume_after_failed_execution_if_configured()
            return

        robot_x, robot_y = robot_position
        standoff = max(
            float(approach_standoff_distance),
            MIN_DISTANCE_EPSILON_M,
        )

        values = (
            target_x,
            target_y,
            observation_x,
            observation_y,
            robot_x,
            robot_y,
            standoff,
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error(
                f"Approach geometry contains non-finite values: {values}"
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_TARGET_POSE_INVALID,
            )
            self.resume_after_failed_execution_if_configured()
            return

        axis_x, axis_y, axis_source = self.get_approach_axis(
            policy=policy,
            target_x=target_x,
            target_y=target_y,
            observation_x=observation_x,
            observation_y=observation_y,
            robot_x=robot_x,
            robot_y=robot_y,
        )

        if axis_x is None or axis_y is None:
            self.get_logger().error(
                "Could not calculate a valid approach axis for the tag."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_TARGET_POSE_INVALID,
            )
            self.resume_after_failed_execution_if_configured()
            return

        # The highest-confidence observation pose defines the only valid
        # free-space side of the tag. Never generate the opposite side.
        goal_x = target_x + standoff * axis_x
        goal_y = target_y + standoff * axis_y

        goal_yaw = math.atan2(
            target_y - goal_y,
            target_x - goal_x,
        )

        candidates: List[Dict[str, Any]] = [
            {
                "name": "observation_side",
                "x": goal_x,
                "y": goal_y,
                "yaw": goal_yaw,
                "pose": self.make_pose_stamped(
                    goal_x,
                    goal_y,
                    goal_yaw,
                ),
                "valid": False,
                "path_length": math.inf,
            }
        ]

        self.execution_generation += 1
        generation = self.execution_generation

        self.execution = {
            "generation": generation,
            "policy": copy.deepcopy(policy),
            "stage": self.STAGE_PLANNING,
            "target_x": target_x,
            "target_y": target_y,
            "energy_before": self.get_current_energy(),
            "reward_before": self.get_current_reward(),
            "arrival_energy": None,
            "arrival_reward": None,
            "interaction_deadline_ns": None,
            "selected_candidate_name": None,
        }
        self.candidate_plan_candidates = candidates
        self.candidate_plan_index = 0

        self.get_logger().info(
            "Checking observation-side tag approach with ComputePathToPose: "
            f"policy={int(policy.policy_id)} '{policy.policy_name}', "
            f"tag_id={int(policy.target_tag_id)}, "
            f"tag=({target_x:.3f}, {target_y:.3f}), "
            f"observation=({observation_x:.3f}, {observation_y:.3f}), "
            f"standoff={standoff:.3f} m, "
            f"axis=({axis_x:.3f}, {axis_y:.3f}) from {axis_source}; "
            f"goal=({goal_x:.3f}, {goal_y:.3f})"
        )

        if not self.compute_path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 ComputePathToPose action server is unavailable."
            )
            self.finish_execution(
                generation,
                success=False,
                failure_reason=constants.FAILURE_PATH_UNAVAILABLE,
                resume_exploration=self.resume_exploration_after_failure,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
            )
            return

        self.request_next_candidate_path(generation)

    def request_next_candidate_path(self, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_PLANNING)
        if context is None:
            return

        if self.candidate_plan_index >= len(self.candidate_plan_candidates):
            self.select_and_execute_candidate(generation)
            return

        index = self.candidate_plan_index
        candidate = self.candidate_plan_candidates[index]
        candidate["pose"].header.stamp = self.get_clock().now().to_msg()

        path_goal = ComputePathToPose.Goal()
        path_goal.goal = candidate["pose"]
        path_goal.planner_id = ""
        path_goal.use_start = False

        self.get_logger().info(
            f"Planning candidate {index + 1}/"
            f"{len(self.candidate_plan_candidates)} "
            f"{candidate['name']}: "
            f"({candidate['x']:.3f}, {candidate['y']:.3f}, "
            f"yaw={candidate['yaw']:.3f})"
        )

        send_future = self.compute_path_client.send_goal_async(path_goal)
        send_future.add_done_callback(
            lambda future, g=generation, i=index:
            self.candidate_path_goal_response_callback(future, g, i)
        )

    def candidate_path_goal_response_callback(
        self,
        future,
        generation: int,
        index: int,
    ) -> None:
        if self.get_execution(generation, self.STAGE_PLANNING) is None:
            return

        try:
            goal_handle = future.result()
        except Exception as ex:  # noqa: BLE001 - ROS future exceptions vary
            self.get_logger().error(
                f"Candidate {index} path request failed: {ex}"
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="request_exception",
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(
                f"Candidate {index} ComputePathToPose goal was rejected."
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="goal_rejected",
            )
            return

        self.active_path_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, g=generation, i=index:
            self.candidate_path_result_callback(future, g, i)
        )

    def candidate_path_result_callback(
        self,
        future,
        generation: int,
        index: int,
    ) -> None:
        if self.get_execution(generation, self.STAGE_PLANNING) is None:
            return

        self.active_path_goal_handle = None

        try:
            wrapped_result = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(
                f"Candidate {index} path result failed: {ex}"
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="result_exception",
            )
            return

        succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        path = wrapped_result.result.path
        path_has_poses = len(path.poses) > 0
        valid = bool(succeeded and path_has_poses)
        path_length = self.compute_path_length(path) if valid else math.inf
        reason = (
            "reachable"
            if valid
            else f"status_{wrapped_result.status}_empty_{not path_has_poses}"
        )

        self.record_candidate_path_result(
            generation,
            index,
            valid=valid,
            path_length=path_length,
            reason=reason,
        )

    def record_candidate_path_result(
        self,
        generation: int,
        index: int,
        valid: bool,
        path_length: float,
        reason: str,
    ) -> None:
        if self.get_execution(generation, self.STAGE_PLANNING) is None:
            return

        if index >= len(self.candidate_plan_candidates):
            return

        candidate = self.candidate_plan_candidates[index]
        candidate["valid"] = bool(valid)
        candidate["path_length"] = float(path_length)

        if valid:
            self.get_logger().info(
                f"Candidate {candidate['name']} is reachable; "
                f"path_length={path_length:.3f} m."
            )
        else:
            self.get_logger().warn(
                f"Candidate {candidate['name']} is not usable: {reason}."
            )

        self.candidate_plan_index = index + 1
        self.request_next_candidate_path(generation)

    def select_and_execute_candidate(self, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_PLANNING)
        if context is None:
            return

        self.active_path_goal_handle = None

        candidate = self.candidate_plan_candidates[0]

        if not candidate["valid"]:
            self.get_logger().error(
                "The observation-side tag approach is not reachable."
            )
            self.finish_execution(
                generation,
                success=False,
                failure_reason=constants.FAILURE_TARGET_UNREACHABLE,
                resume_exploration=self.resume_exploration_after_failure,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
            )
            return

        context["selected_candidate_name"] = "observation_side"

        self.get_logger().info(
            "Selected observation-side approach: "
            f"goal=({candidate['x']:.3f}, {candidate['y']:.3f}, "
            f"yaw={candidate['yaw']:.3f}), "
            f"path_length={candidate['path_length']:.3f} m."
        )

        self.send_navigation_goal(generation, candidate)