"""Execute eMDB-selected Robotino foraging policies safely.

Behavioral contract
-------------------
* CONTINUE_EXPLORING and SEARCH_FOR_ENERGY enable exploration.
* INSPECT_VISIBLE_TAG only acknowledges that perception/memory stored a tag.
  It never disables exploration, cancels navigation, or approaches the tag.
* RETURN_TO_BEST_ENERGY_BANK approaches the remembered energy tag, then waits
  for a real energy increase before reporting semantic success.
* GOAL approaches the remembered goal tag, then waits for a real reward/goal
  confirmation before reporting semantic success.
* Stale Nav2 callbacks are ignored with an execution-generation token, so a
  result from an old/cancelled energy goal cannot be attributed to a newer
  goal policy.

Policy priority still belongs to the eMDB policy selector. This executor adds
one defensive priority rule: an active energy-return policy is never preempted
by a goal policy.
"""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from robotino_emdb_actuation.navigation_geometry import (
    yaw_to_quaternion_z_w,
)
from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)

MIN_DISTANCE_EPSILON_M = 0.001


class RobotinoPolicyExecutor(Node):
    """Actuation layer between the eMDB policy selector and Nav2."""

    POLICY_CONTINUE_EXPLORING = 0
    POLICY_INSPECT_VISIBLE_TAG = 1
    POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
    POLICY_SEARCH_FOR_ENERGY = 3
    POLICY_GOAL = 4

    POLICY_TOPIC = "/robotino/emdb/selected_policy"
    FORAGING_TOPIC = "/robotino/emdb/foraging_state"
    OUTCOME_TOPIC = "/robotino/emdb/policy_outcome"

    CMD_VEL_TOPIC = "/cmd_vel"
    EXPLORATION_ENABLE_TOPIC = "/robotino/emdb/frontier_exploration_enable"

    NAV2_ACTION_NAME = "navigate_to_pose"
    COMPUTE_PATH_ACTION_NAME = "compute_path_to_pose"
    MAP_FRAME = "map"

    STAGE_PLANNING = "planning"
    STAGE_NAVIGATING = "navigating"
    STAGE_WAITING_INTERACTION = "waiting_interaction"

    def __init__(self) -> None:
        super().__init__("robotino_policy_executor")

        # Frames and action behavior.
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("enable_nav2_execution", False)
        self.declare_parameter("minimum_goal_interval_s", 0.0)

        # Tag approach geometry. These are center-of-base distances from the
        # tag/wall. Robotino radius is about 0.20 m; 0.65 m leaves room for
        # inflation and localization error.
        self.declare_parameter("energy_approach_standoff_m", 0.65)
        self.declare_parameter("goal_approach_standoff_m", 0.65)

        # Semantic completion must be confirmed by state change, not merely by
        # Nav2 reaching a standoff pose.
        self.declare_parameter("energy_success_delta", 0.01)
        self.declare_parameter("reward_success_delta", 0.01)
        self.declare_parameter("interaction_timeout_s", 12.0)
        self.declare_parameter("interaction_check_period_s", 0.20)

        # After charging, the requested behavior is to continue exploration.
        self.declare_parameter("resume_exploration_after_energy", True)
        self.declare_parameter("resume_exploration_after_failure", True)

        self.robot_base_frame = str(
            self.get_parameter("robot_base_frame").value
        )
        self.enable_nav2_execution = bool(
            self.get_parameter("enable_nav2_execution").value
        )
        self.minimum_goal_interval_s = max(
            0.0,
            float(self.get_parameter("minimum_goal_interval_s").value),
        )
        self.energy_approach_standoff_m = max(
            MIN_DISTANCE_EPSILON_M,
            float(self.get_parameter("energy_approach_standoff_m").value),
        )
        self.goal_approach_standoff_m = max(
            MIN_DISTANCE_EPSILON_M,
            float(self.get_parameter("goal_approach_standoff_m").value),
        )
        self.energy_success_delta = max(
            0.0,
            float(self.get_parameter("energy_success_delta").value),
        )
        self.reward_success_delta = max(
            0.0,
            float(self.get_parameter("reward_success_delta").value),
        )
        self.interaction_timeout_s = max(
            0.1,
            float(self.get_parameter("interaction_timeout_s").value),
        )
        self.interaction_check_period_s = max(
            0.05,
            float(self.get_parameter("interaction_check_period_s").value),
        )
        self.resume_exploration_after_energy = bool(
            self.get_parameter("resume_exploration_after_energy").value
        )
        self.resume_exploration_after_failure = bool(
            self.get_parameter("resume_exploration_after_failure").value
        )

        self.map_frame = self.MAP_FRAME

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_foraging_state: Optional[RobotinoForagingState] = None
        self.last_goal_time = None

        # A single generation protects ComputePathToPose and NavigateToPose
        # callbacks. Incrementing it invalidates every callback belonging to an
        # old or cancelled execution.
        self.execution_generation = 0
        self.execution: Optional[Dict[str, Any]] = None

        self.active_path_goal_handle = None
        self.active_navigation_goal_handle = None

        self.candidate_plan_candidates: List[Dict[str, Any]] = []
        self.candidate_plan_index = 0

        self.exploration_enable_publisher = self.create_publisher(
            Bool,
            self.EXPLORATION_ENABLE_TOPIC,
            10,
        )
        self.outcome_publisher = self.create_publisher(
            RobotinoPolicyOutcome,
            self.OUTCOME_TOPIC,
            10,
        )
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.CMD_VEL_TOPIC,
            10,
        )

        self.policy_subscriber = self.create_subscription(
            RobotinoSelectedPolicy,
            self.POLICY_TOPIC,
            self.policy_callback,
            10,
        )
        self.foraging_subscriber = self.create_subscription(
            RobotinoForagingState,
            self.FORAGING_TOPIC,
            self.foraging_callback,
            10,
        )

        self.nav2_client = ActionClient(
            self,
            NavigateToPose,
            self.NAV2_ACTION_NAME,
        )
        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            self.COMPUTE_PATH_ACTION_NAME,
        )

        self.interaction_timer = self.create_timer(
            self.interaction_check_period_s,
            self.interaction_timer_callback,
        )

        self.get_logger().info(
            "Robotino policy executor ready. "
            f"Nav2 execution={'enabled' if self.enable_nav2_execution else 'disabled'}; "
            "visible-tag inspection is memory-only."
        )

    # ------------------------------------------------------------------
    # Inputs and high-level policy dispatch
    # ------------------------------------------------------------------

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        self.latest_foraging_state = msg

        if self.is_waiting_for_interaction():
            self.check_interaction_completion()

    def policy_callback(self, msg: RobotinoSelectedPolicy) -> None:
        if not bool(msg.valid) or not bool(msg.execute_now):
            return

        policy_id = int(msg.policy_id)

        if policy_id == self.POLICY_CONTINUE_EXPLORING:
            self.execute_continue_exploring(msg)
        elif policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            self.execute_inspect_visible_tag(msg)
        elif policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.execute_return_to_best_energy_bank(msg)
        elif policy_id == self.POLICY_SEARCH_FOR_ENERGY:
            self.execute_search_for_energy(msg)
        elif policy_id == self.POLICY_GOAL:
            self.execute_goal(msg)
        else:
            self.get_logger().warn(
                f"Ignoring unknown policy_id={policy_id}, "
                f"policy_name='{msg.policy_name}'."
            )

    def execute_continue_exploring(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_continue_exploring",
            publish_outcome=True,
        )
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="continuing_exploration",
        )

    def execute_search_for_energy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_search_for_energy",
            publish_outcome=True,
        )
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="searching_for_energy",
        )

    def execute_inspect_visible_tag(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        """Acknowledge perception/memory without changing robot motion.

        The perception/memory layer is responsible for saving the tag. This
        executor intentionally does not cancel exploration, cancel an energy
        return, disable exploration, or issue a Nav2 goal here.
        """
        active_description = self.describe_active_execution()
        self.get_logger().info(
            f"Observed tag {int(policy.target_tag_id)} saved by memory layer; "
            f"motion unchanged ({active_description})."
        )
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="tag_observed_and_saved_no_motion",
        )

    def execute_return_to_best_energy_bank(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated energy-return policy ignored; execution is already active."
            )
            return

        # Energy has priority over goal. If a goal approach is running, close
        # it as preempted and begin the energy return immediately.
        if self.execution is not None:
            self.cancel_current_execution(
                reason="preempted_by_higher_priority_energy_policy",
                publish_outcome=True,
            )

        self.start_tag_navigation(
            policy,
            approach_standoff_distance=self.energy_approach_standoff_m,
        )

    def execute_goal(self, policy: RobotinoSelectedPolicy) -> None:
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated goal policy ignored; execution is already active."
            )
            return

        # Defensive priority guard. A goal policy may not interrupt returning
        # to an energy bank or waiting for charging confirmation.
        if self.active_policy_id() == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.get_logger().warn(
                "Goal policy deferred because an energy-return policy is active. "
                "The goal remains stored for later."
            )
            return

        if self.execution is not None:
            self.cancel_current_execution(
                reason="preempted_by_new_goal_policy",
                publish_outcome=True,
            )

        self.start_tag_navigation(
            policy,
            approach_standoff_distance=self.goal_approach_standoff_m,
        )

    # ------------------------------------------------------------------
    # Navigation planning and execution
    # ------------------------------------------------------------------

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
                started=False,
                success=False,
                status="policy_does_not_use_nav2",
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
                started=False,
                success=False,
                status="dry_run_nav2_disabled",
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
                started=False,
                success=False,
                status="no_current_robot_tf_pose",
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
                started=False,
                success=False,
                status="invalid_approach_geometry",
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
                started=False,
                success=False,
                status="invalid_tag_approach_axis",
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
                status="compute_path_server_unavailable",
                resume_exploration=self.resume_exploration_after_failure,
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
                status="observation_side_not_reachable",
                resume_exploration=self.resume_exploration_after_failure,
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

    def send_navigation_goal(
        self,
        generation: int,
        selected: Dict[str, Any],
    ) -> None:
        context = self.get_execution(generation, self.STAGE_PLANNING)
        if context is None:
            return

        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 NavigateToPose action server is unavailable."
            )
            self.finish_execution(
                generation,
                success=False,
                status="nav2_server_unavailable",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = selected["pose"]
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        context["stage"] = self.STAGE_NAVIGATING
        self.last_goal_time = self.get_clock().now()

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future, g=generation:
            self.nav2_goal_response_callback(future, g)
        )

    def nav2_goal_response_callback(self, future, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            return

        try:
            goal_handle = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f"Failed to send Nav2 goal: {ex}")
            self.finish_execution(
                generation,
                success=False,
                status="nav2_goal_send_failed",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Nav2 approach goal was rejected.")
            self.finish_execution(
                generation,
                success=False,
                status="nav2_goal_rejected",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        self.active_navigation_goal_handle = goal_handle
        self.get_logger().info("Nav2 approach goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, g=generation:
            self.nav2_result_callback(result, g)
        )

    def nav2_result_callback(self, future, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            # This is the critical stale-callback protection. A result from a
            # cancelled/older policy is never applied to the current policy.
            self.get_logger().debug(
                f"Ignoring stale Nav2 result for generation {generation}."
            )
            return

        self.active_navigation_goal_handle = None

        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to receive Nav2 result: {ex}"
            )
            status = -1

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.finish_execution(
                generation,
                success=False,
                status=f"nav2_goal_finished_status_{status}",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        self.stop_robot()
        self.enter_interaction_wait(generation)

    # ------------------------------------------------------------------
    # Semantic interaction confirmation
    # ------------------------------------------------------------------

    def enter_interaction_wait(self, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            return

        policy = context["policy"]
        policy_id = int(policy.policy_id)

        context["stage"] = self.STAGE_WAITING_INTERACTION
        if self.latest_foraging_state is None:
            context["arrival_energy"] = None
            context["arrival_reward"] = None
            context["arrival_goal_complete"] = False
        else:
            context["arrival_energy"] = self.get_current_energy()
            context["arrival_reward"] = self.get_current_reward()
            context["arrival_goal_complete"] = (
                self.foraging_state_reports_goal_complete()
            )

        context["interaction_deadline_ns"] = (
            self.get_clock().now().nanoseconds
            + int(self.interaction_timeout_s * 1e9)
        )

        candidate_name = context.get("selected_candidate_name") or "candidate"

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            status = (
                "energy_approach_pose_reached_waiting_for_charge_"
                f"{candidate_name}"
            )
        elif policy_id == self.POLICY_GOAL:
            status = (
                "goal_approach_pose_reached_waiting_for_reward_"
                f"{candidate_name}"
            )
        else:
            # Only energy and goal policies should use tag navigation now.
            self.finish_execution(
                generation,
                success=True,
                status=f"tag_approach_pose_reached_{candidate_name}",
                resume_exploration=False,
            )
            return

        self.get_logger().info(status)

        # This is intentionally unfinished and unsuccessful. Reaching a Nav2
        # standoff pose is not the same as charging or completing the goal.
        self.publish_policy_outcome(
            policy,
            started=True,
            finished=False,
            success=False,
            status=status,
            energy_before=float(context["energy_before"]),
            reward_before=float(context["reward_before"]),
        )

        self.check_interaction_completion()

    def check_interaction_completion(self) -> None:
        context = self.get_waiting_context()
        if context is None:
            return

        policy = context["policy"]
        policy_id = int(policy.policy_id)
        generation = int(context["generation"])

        if self.latest_foraging_state is None:
            return

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            if context["arrival_energy"] is None:
                # Establish a valid baseline on the first state sample instead
                # of comparing a real energy value against the fallback 0.0.
                context["arrival_energy"] = self.get_current_energy()
                return

            arrival_energy = float(context["arrival_energy"])
            current_energy = self.get_current_energy()
            energy_delta = current_energy - arrival_energy

            if energy_delta >= self.energy_success_delta:
                self.get_logger().info(
                    "Charging confirmed: "
                    f"energy increased by {energy_delta:.3f}."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    status="energy_charge_confirmed",
                    resume_exploration=self.resume_exploration_after_energy,
                )

        elif policy_id == self.POLICY_GOAL:
            if context["arrival_reward"] is None:
                # Establish both baselines on the first state sample. This
                # prevents a late first state message from looking like reward.
                context["arrival_reward"] = self.get_current_reward()
                context["arrival_goal_complete"] = (
                    self.foraging_state_reports_goal_complete()
                )
                return

            arrival_reward = float(context["arrival_reward"])
            current_reward = self.get_current_reward()
            reward_delta = current_reward - arrival_reward
            goal_flag_transition = (
                not bool(context.get("arrival_goal_complete", False))
                and self.foraging_state_reports_goal_complete()
            )

            if (
                reward_delta >= self.reward_success_delta
                or goal_flag_transition
            ):
                self.get_logger().info(
                    "Goal interaction confirmed: "
                    f"reward increased by {reward_delta:.3f}."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    status="goal_interaction_confirmed",
                    resume_exploration=False,
                )

    def interaction_timer_callback(self) -> None:
        context = self.get_waiting_context()
        if context is None:
            return

        self.check_interaction_completion()

        # check_interaction_completion() may have finished and cleared it.
        context = self.get_waiting_context()
        if context is None:
            return

        deadline_ns = context.get("interaction_deadline_ns")
        if deadline_ns is None:
            return

        if self.get_clock().now().nanoseconds < int(deadline_ns):
            return

        policy_id = int(context["policy"].policy_id)
        generation = int(context["generation"])

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            status = "energy_interaction_timeout_no_charge_detected"
        elif policy_id == self.POLICY_GOAL:
            status = "goal_interaction_timeout_no_reward_detected"
        else:
            status = "interaction_timeout"

        self.get_logger().warn(status)
        self.finish_execution(
            generation,
            success=False,
            status=status,
            resume_exploration=self.resume_exploration_after_failure,
        )

    def foraging_state_reports_goal_complete(self) -> bool:
        """Use an explicit completion flag when the message provides one.

        The currently used message is known to contain total_reward. This
        introspection also supports future interface versions without making
        this node depend on a field that may not yet exist.
        """
        if self.latest_foraging_state is None:
            return False

        for field_name in (
            "goal_reached",
            "goal_achieved",
            "task_complete",
            "task_completed",
        ):
            if hasattr(self.latest_foraging_state, field_name):
                return bool(getattr(self.latest_foraging_state, field_name))

        return False

    # ------------------------------------------------------------------
    # Execution lifecycle and stale-callback safety
    # ------------------------------------------------------------------

    def finish_execution(
        self,
        generation: int,
        success: bool,
        status: str,
        resume_exploration: bool,
    ) -> None:
        context = self.get_execution(generation)
        if context is None:
            return

        policy = context["policy"]
        energy_before = float(context["energy_before"])
        reward_before = float(context["reward_before"])

        # Invalidate callbacks before publishing or enabling another behavior.
        self.execution_generation += 1
        self.execution = None
        self.active_path_goal_handle = None
        self.active_navigation_goal_handle = None
        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0

        self.publish_policy_outcome(
            policy,
            started=True,
            finished=True,
            success=success,
            status=status,
            energy_before=energy_before,
            reward_before=reward_before,
        )

        if resume_exploration:
            self.set_exploration_enabled(True)
            self.get_logger().info(
                "Exploration re-enabled after policy completion."
            )

    def cancel_current_execution(
        self,
        reason: str,
        publish_outcome: bool,
    ) -> None:
        old_context = self.execution

        # Invalidate all in-flight callbacks first.
        self.execution_generation += 1
        self.execution = None

        if self.active_path_goal_handle is not None:
            try:
                self.active_path_goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    f"Failed to cancel path-planning goal: {ex}"
                )
            self.active_path_goal_handle = None

        if self.active_navigation_goal_handle is not None:
            try:
                self.active_navigation_goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    f"Failed to cancel navigation goal: {ex}"
                )
            self.active_navigation_goal_handle = None

        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0

        if old_context is None:
            return

        self.stop_robot()
        old_policy = old_context["policy"]
        self.get_logger().info(
            f"Cancelled policy {int(old_policy.policy_id)} "
            f"'{old_policy.policy_name}': {reason}."
        )

        if publish_outcome:
            self.publish_policy_outcome(
                old_policy,
                started=True,
                finished=True,
                success=False,
                status=reason,
                energy_before=float(old_context["energy_before"]),
                reward_before=float(old_context["reward_before"]),
            )

    def get_execution(
        self,
        generation: int,
        required_stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.execution is None:
            return None

        if int(self.execution["generation"]) != int(generation):
            return None

        if required_stage is not None:
            if self.execution.get("stage") != required_stage:
                return None

        return self.execution

    def get_waiting_context(self) -> Optional[Dict[str, Any]]:
        if self.execution is None:
            return None
        if self.execution.get("stage") != self.STAGE_WAITING_INTERACTION:
            return None
        return self.execution

    def is_waiting_for_interaction(self) -> bool:
        return self.get_waiting_context() is not None

    def active_policy_id(self) -> Optional[int]:
        if self.execution is None:
            return None
        return int(self.execution["policy"].policy_id)

    def is_same_policy_already_active(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> bool:
        if self.execution is None:
            return False

        active_policy = self.execution["policy"]
        return (
            int(active_policy.policy_id) == int(policy.policy_id)
            and int(active_policy.target_tag_id) == int(policy.target_tag_id)
            and math.isclose(
                float(active_policy.target_x_map),
                float(policy.target_x_map),
                abs_tol=0.02,
            )
            and math.isclose(
                float(active_policy.target_y_map),
                float(policy.target_y_map),
                abs_tol=0.02,
            )
        )

    def describe_active_execution(self) -> str:
        if self.execution is None:
            return "no active navigation policy"

        policy = self.execution["policy"]
        return (
            f"policy_id={int(policy.policy_id)}, "
            f"stage={self.execution.get('stage', 'unknown')}"
        )

    # ------------------------------------------------------------------
    # Geometry and ROS helpers
    # ------------------------------------------------------------------

    def get_robot_position_from_tf(self) -> Optional[Tuple[float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            return (
                float(tf_msg.transform.translation.x),
                float(tf_msg.transform.translation.y),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"Could not get current robot pose from TF "
                f"{self.map_frame} <- {self.robot_base_frame}: {ex}"
            )
            return None

    def get_approach_axis(
        self,
        policy: RobotinoSelectedPolicy,
        target_x: float,
        target_y: float,
        observation_x: float,
        observation_y: float,
        robot_x: float,
        robot_y: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Return a unit vector from the tag toward known free space."""
        for x_field, y_field in (
            ("target_normal_x_map", "target_normal_y_map"),
            ("normal_x_map", "normal_y_map"),
        ):
            if not hasattr(policy, x_field) or not hasattr(policy, y_field):
                continue

            nx = float(getattr(policy, x_field))
            ny = float(getattr(policy, y_field))
            magnitude = math.hypot(nx, ny)

            if (
                math.isfinite(nx)
                and math.isfinite(ny)
                and magnitude > MIN_DISTANCE_EPSILON_M
            ):
                return (
                    nx / magnitude,
                    ny / magnitude,
                    f"{x_field}/{y_field}",
                )

        # The pose from which the tag was actually observed is a known-free
        # side and is safer than interpreting an arbitrary Euler yaw as the
        # wall-normal direction.
        dx = observation_x - target_x
        dy = observation_y - target_y
        magnitude = math.hypot(dx, dy)
        if magnitude > MIN_DISTANCE_EPSILON_M:
            return (
                dx / magnitude,
                dy / magnitude,
                "last_successful_observation_pose",
            )

        # Last-resort fallback if observation and target coordinates coincide.
        dx = robot_x - target_x
        dy = robot_y - target_y
        magnitude = math.hypot(dx, dy)
        if magnitude > MIN_DISTANCE_EPSILON_M:
            return (
                dx / magnitude,
                dy / magnitude,
                "current_robot_pose_fallback",
            )

        return None, None, "unavailable"

    def make_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        qz, qw = yaw_to_quaternion_z_w(yaw)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    @staticmethod
    def compute_path_length(path) -> float:
        if len(path.poses) < 2:
            return 0.0

        total = 0.0
        previous = path.poses[0].pose.position

        for pose_stamped in path.poses[1:]:
            current = pose_stamped.pose.position
            total += math.hypot(
                float(current.x) - float(previous.x),
                float(current.y) - float(previous.y),
            )
            previous = current

        return total

    def goal_interval_ok(self) -> bool:
        if self.last_goal_time is None:
            return True

        elapsed_s = (
            self.get_clock().now().nanoseconds
            - self.last_goal_time.nanoseconds
        ) / 1e9
        return elapsed_s >= self.minimum_goal_interval_s

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def publish_simple_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        started: bool,
        success: bool,
        status: str,
    ) -> None:
        energy_before = self.get_current_energy()
        reward_before = self.get_current_reward()
        self.publish_policy_outcome(
            policy,
            started=started,
            finished=True,
            success=success,
            status=status,
            energy_before=energy_before,
            reward_before=reward_before,
        )

    def publish_policy_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        started: bool,
        finished: bool,
        success: bool,
        status: str,
        energy_before: float,
        reward_before: float,
    ) -> None:
        outcome = RobotinoPolicyOutcome()
        outcome.header.stamp = self.get_clock().now().to_msg()
        outcome.header.frame_id = self.map_frame
        outcome.valid = True

        outcome.policy_id = int(policy.policy_id)
        outcome.policy_name = str(policy.policy_name)
        outcome.started = bool(started)
        outcome.finished = bool(finished)
        outcome.success = bool(success)
        outcome.status = str(status)
        outcome.target_tag_id = int(policy.target_tag_id)

        energy_after = self.get_current_energy()
        reward_after = self.get_current_reward()

        outcome.energy_before = float(energy_before)
        outcome.energy_after = float(energy_after)
        outcome.energy_delta = float(energy_after - energy_before)

        outcome.reward_before = float(reward_before)
        outcome.reward_after = float(reward_after)
        outcome.reward_delta = float(reward_after - reward_before)

        self.outcome_publisher.publish(outcome)

    def get_current_energy(self) -> float:
        if self.latest_foraging_state is None:
            return 0.0
        return float(self.latest_foraging_state.robot_energy)

    def get_current_reward(self) -> float:
        if self.latest_foraging_state is None:
            return 0.0
        return float(self.latest_foraging_state.total_reward)

    def set_exploration_enabled(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.exploration_enable_publisher.publish(msg)

    def stop_robot(self) -> None:
        self.cmd_vel_publisher.publish(Twist())

    def resume_after_failed_execution_if_configured(self) -> None:
        if self.resume_exploration_after_failure:
            self.set_exploration_enabled(True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotinoPolicyExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cancel_current_execution(
            reason="node_shutdown",
            publish_outcome=False,
        )
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()