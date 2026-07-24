"""Execute eMDB-selected Robotino foraging policies safely.

Behavioral contract
-------------------
* CONTINUE_EXPLORING and SEARCH_FOR_ENERGY enable exploration.
* INSPECT_VISIBLE_TAG only acknowledges that perception/memory stored a tag.
  It never disables exploration, cancels navigation, or approaches the tag.
* RETURN_TO_BEST_ENERGY_BANK approaches the remembered energy tag and
  confirms that charging started. The visit succeeds when either the recovery
  threshold is reached or the current bank is drained after transferring
  energy, allowing eMDB to select another bank. A renewable bank may regenerate
  later, but regenerated energy is not consumed during the same visit.
* GOAL approaches the remembered goal tag, then waits for a real reward/goal
  confirmation before reporting semantic success.
* Stale Nav2 callbacks are ignored with an execution-generation token, so a
  result from an old/cancelled energy goal cannot be attributed to a newer
  goal policy.

Policy priority still belongs to the eMDB policy selector. This executor adds
one defensive priority rule: an active energy-return policy is never preempted
by a goal policy.

This node's behavior is split across mixins, one per concern, all combined
onto RobotinoPolicyExecutor below so they share a single `self`:

* policy_dispatch      - subscriber callbacks and per-policy routing
* navigation_planning  - ComputePathToPose candidate evaluation
* navigation_execution - NavigateToPose goal send/result handling
* interaction_waiting  - post-arrival energy/reward confirmation
* execution_lifecycle  - generation-token finish/cancel and state queries
* geometry_helpers     - TF lookup, approach-axis math, pose/path helpers
* outcome_publishing    - RobotinoPolicyOutcome publishing and simple I/O
"""

from collections import deque
import math
import random
from typing import Any, Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from std_msgs.msg import Bool, Float32, Int32
from tf2_ros import Buffer, TransformListener

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)

from . import constants
from .execution_lifecycle import ExecutionLifecycleMixin
from .geometry_helpers import GeometryHelpersMixin
from .interaction_waiting import InteractionWaitingMixin
from .mapped_wandering import MappedWanderingMixin
from .navigation_execution import NavigationExecutionMixin
from .navigation_planning import NavigationPlanningMixin
from .outcome_publishing import OutcomePublishingMixin
from .policy_dispatch import PolicyDispatchMixin

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

MIN_DISTANCE_EPSILON_M = 0.001


class RobotinoPolicyExecutor(
    Node,
    PolicyDispatchMixin,
    MappedWanderingMixin,
    NavigationPlanningMixin,
    NavigationExecutionMixin,
    InteractionWaitingMixin,
    ExecutionLifecycleMixin,
    GeometryHelpersMixin,
    OutcomePublishingMixin,
):
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
    RECHARGE_TARGET_TOPIC = "/robotino/emdb/recharge_target"

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
        self.declare_parameter("navigation_timeout_s", 90.0)

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
        self.declare_parameter("energy_progress_delta", 0.001)
        self.declare_parameter("resource_empty_epsilon", 0.01)
        # Start the semantic interaction once the matching visible bank is
        # physically within transfer range, even if Nav2 has not yet satisfied
        # the exact standoff pose/yaw goal.
        self.declare_parameter("energy_interaction_distance_m", 1.20)
        self.declare_parameter(
            "recharge_target_topic",
            self.RECHARGE_TARGET_TOPIC,
        )

        # After charging, the requested behavior is to continue exploration.
        self.declare_parameter("resume_exploration_after_energy", True)
        self.declare_parameter("resume_exploration_after_failure", True)

        # Post-mapping wandering uses the existing map, ComputePathToPose, and
        # NavigateToPose clients owned by this executor.
        self.declare_parameter(
            "mapping_complete_topic",
            "/frontier_exploration/mapping_complete",
        )
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter(
            "exploration_satisfaction_topic",
            "/robotino/emdb/satisfaction/exploration",
        )
        self.declare_parameter("wander_energy_threshold", 0.35)
        self.declare_parameter("resume_energy_threshold", 0.70)
        self.declare_parameter("minimum_energy_bank_score", 0.0)
        self.declare_parameter("minimum_energy_bank_worthiness", 0.10)
        self.declare_parameter("wander_min_distance_m", 1.25)
        self.declare_parameter("wander_max_distance_m", 5.0)
        self.declare_parameter("wander_clearance_m", 0.40)
        self.declare_parameter("wander_free_threshold", 10)
        self.declare_parameter("wander_candidate_samples", 300)
        self.declare_parameter("wander_path_checks", 12)
        self.declare_parameter("wander_recent_goal_count", 20)
        self.declare_parameter("wander_recent_goal_radius_m", 1.0)
        self.declare_parameter("wander_random_seed", 0)
        self.declare_parameter(
            "wandering_active_topic",
            "/robotino/emdb/wandering_active",
        )
        self.declare_parameter(
            "wander_goal_topic",
            "/robotino/emdb/wander_goal",
        )

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
        self.navigation_timeout_s = max(
            5.0,
            float(self.get_parameter("navigation_timeout_s").value),
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
        self.energy_progress_delta = max(
            1e-6,
            float(self.get_parameter("energy_progress_delta").value),
        )
        self.resource_empty_epsilon = max(
            0.0,
            float(self.get_parameter("resource_empty_epsilon").value),
        )
        self.energy_interaction_distance_m = max(
            0.05,
            float(
                self.get_parameter("energy_interaction_distance_m").value
            ),
        )
        self.recharge_target_topic = str(
            self.get_parameter("recharge_target_topic").value
        )
        self.resume_exploration_after_energy = bool(
            self.get_parameter("resume_exploration_after_energy").value
        )
        self.resume_exploration_after_failure = bool(
            self.get_parameter("resume_exploration_after_failure").value
        )

        self.mapping_complete_topic = str(
            self.get_parameter("mapping_complete_topic").value
        )
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.exploration_satisfaction_topic = str(
            self.get_parameter("exploration_satisfaction_topic").value
        )
        self.wander_energy_threshold = max(
            0.0,
            min(
                1.0,
                float(self.get_parameter("wander_energy_threshold").value),
            ),
        )
        self.resume_energy_threshold = max(
            self.wander_energy_threshold,
            min(
                1.0,
                float(self.get_parameter("resume_energy_threshold").value),
            ),
        )
        self.minimum_energy_bank_score = max(
            0.0,
            float(self.get_parameter("minimum_energy_bank_score").value),
        )
        self.minimum_energy_bank_worthiness = max(
            0.0,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "minimum_energy_bank_worthiness"
                    ).value
                ),
            ),
        )
        self.wander_min_distance_m = max(
            0.0,
            float(self.get_parameter("wander_min_distance_m").value),
        )
        self.wander_max_distance_m = max(
            self.wander_min_distance_m + 0.10,
            float(self.get_parameter("wander_max_distance_m").value),
        )
        self.wander_clearance_m = max(
            0.0,
            float(self.get_parameter("wander_clearance_m").value),
        )
        self.wander_free_threshold = int(
            self.get_parameter("wander_free_threshold").value
        )
        self.wander_candidate_samples = max(
            1,
            int(self.get_parameter("wander_candidate_samples").value),
        )
        self.wander_path_checks = max(
            1,
            int(self.get_parameter("wander_path_checks").value),
        )
        wander_recent_goal_count = max(
            1,
            int(self.get_parameter("wander_recent_goal_count").value),
        )
        self.wander_recent_goal_radius_m = max(
            0.0,
            float(self.get_parameter("wander_recent_goal_radius_m").value),
        )
        wander_random_seed = int(
            self.get_parameter("wander_random_seed").value
        )
        self.wandering_active_topic = str(
            self.get_parameter("wandering_active_topic").value
        )
        self.wander_goal_topic = str(
            self.get_parameter("wander_goal_topic").value
        )

        self.map_frame = self.MAP_FRAME

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_foraging_state: Optional[RobotinoForagingState] = None
        self.energy_recovery_mode = False
        self.energy_mode_initialized = False
        self.frontier_exploration_enabled = False
        self.frontier_exploration_mode: Optional[str] = None
        self.authorized_recharge_target_id = -1
        self.latest_map: Optional[OccupancyGrid] = None
        self.mapping_complete = False
        self.mapping_complete_signal_received = False
        self.wandering_active = False
        self.wander_random = random.Random(wander_random_seed)
        self.recent_wander_goals = deque(maxlen=wander_recent_goal_count)
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
        self.recharge_target_publisher = self.create_publisher(
            Int32,
            self.recharge_target_topic,
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
        self.wandering_active_publisher = self.create_publisher(
            Bool,
            self.wandering_active_topic,
            10,
        )
        self.wander_goal_publisher = self.create_publisher(
            PoseStamped,
            self.wander_goal_topic,
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
        self.mapping_complete_subscriber = self.create_subscription(
            Bool,
            self.mapping_complete_topic,
            self.mapping_complete_callback,
            10,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
        )
        self.exploration_satisfaction_subscriber = self.create_subscription(
            Float32,
            self.exploration_satisfaction_topic,
            self.exploration_satisfaction_callback,
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

        self.set_wandering_active(False, force=True)
        self.set_exploration_enabled(False, force=True)
        self.set_recharge_target(-1, force=True)
        self.get_logger().info(
            "Robotino policy executor ready. "
            f"Nav2 execution={'enabled' if self.enable_nav2_execution else 'disabled'}; "
            "visible-tag inspection is memory-only; "
            f"mapped wandering waits for {self.mapping_complete_topic}=true; "
            f"novelty wandering requires energy>{self.wander_energy_threshold:.2f}; "
            "search_for_energy ignores that lower threshold and runs until "
            f"energy>={self.resume_energy_threshold:.2f} or a worthy bank is known; "
            "frontier motion is guarded by the same energy hysteresis; "
            f"Nav2 watchdog={self.navigation_timeout_s:.1f}s; "
            f"resource empty<={self.resource_empty_epsilon:.3f}; "
            f"energy interaction<={self.energy_interaction_distance_m:.2f}m; "
            f"recharge authorization={self.recharge_target_topic}."
        )

    # ------------------------------------------------------------------
    # Post-arrival recharge authorization and lifecycle
    # ------------------------------------------------------------------

    def set_recharge_target(
        self,
        target_tag_id: int,
        force: bool = False,
    ) -> None:
        """Authorize energy transfer for one tag, or disable it with -1."""
        target_tag_id = int(target_tag_id)
        if (
            not force
            and target_tag_id == self.authorized_recharge_target_id
        ):
            return

        self.authorized_recharge_target_id = target_tag_id
        msg = Int32()
        msg.data = target_tag_id
        self.recharge_target_publisher.publish(msg)

    def finish_execution(
        self,
        generation: int,
        success: bool,
        failure_reason: str,
        resume_exploration: bool,
        navigation_result: int = RobotinoPolicyOutcome.NAV_NOT_USED,
        tag_result: int = RobotinoPolicyOutcome.TAG_NOT_CHECKED,
        detection_confidence: Optional[float] = None,
        observed_tag_pose: Optional[PoseStamped] = None,
        recharge_attempted: bool = False,
        recharge_succeeded: bool = False,
    ) -> None:
        """Clear recharge authorization before closing any execution."""
        self.set_recharge_target(-1)
        ExecutionLifecycleMixin.finish_execution(
            self,
            generation=generation,
            success=success,
            failure_reason=failure_reason,
            resume_exploration=resume_exploration,
            navigation_result=navigation_result,
            tag_result=tag_result,
            detection_confidence=detection_confidence,
            observed_tag_pose=observed_tag_pose,
            recharge_attempted=recharge_attempted,
            recharge_succeeded=recharge_succeeded,
        )

    def cancel_current_execution(
        self,
        reason: str,
        publish_outcome: bool,
    ) -> None:
        """Clear recharge authorization before canceling an execution."""
        self.set_recharge_target(-1)
        ExecutionLifecycleMixin.cancel_current_execution(
            self,
            reason=reason,
            publish_outcome=publish_outcome,
        )


    # ------------------------------------------------------------------
    # Navigation and interaction watchdogs
    # ------------------------------------------------------------------

    def send_navigation_goal(
        self,
        generation: int,
        selected: Dict[str, Any],
    ) -> None:
        """Attach a deadline before delegating goal execution to the mixin."""
        context = self.get_execution(generation, self.STAGE_PLANNING)
        if context is not None:
            context["navigation_deadline_ns"] = (
                self.get_clock().now().nanoseconds
                + int(self.navigation_timeout_s * 1e9)
            )

        NavigationExecutionMixin.send_navigation_goal(
            self,
            generation,
            selected,
        )

    def nav2_goal_response_callback(self, future, generation: int) -> None:
        """Cancel a goal accepted after its execution was already invalidated."""
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            try:
                stale_goal_handle = future.result()
                if stale_goal_handle is not None and stale_goal_handle.accepted:
                    stale_goal_handle.cancel_goal_async()
                    self.get_logger().warn(
                        "Canceled a stale Nav2 goal accepted after execution "
                        f"generation {generation} had already finished."
                    )
            except Exception as ex:  # noqa: BLE001
                self.get_logger().debug(
                    f"Ignored stale Nav2 goal response for generation "
                    f"{generation}: {ex}"
                )
            return

        NavigationExecutionMixin.nav2_goal_response_callback(
            self,
            future,
            generation,
        )

    def interaction_timer_callback(self) -> None:
        """Watch both Nav2 execution and post-arrival semantic interaction."""
        # A tag interaction is a semantic condition, not an exact-pose
        # condition.  Take over from Nav2 as soon as the selected visible bank
        # is inside the configured transfer radius.
        self.try_begin_energy_interaction_from_proximity()
        self.check_navigation_timeout()

        context = self.get_waiting_context()
        if context is None:
            return

        self.check_interaction_completion()

        # Completion may have invalidated and cleared the execution.
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
            recharge_succeeded = bool(
                context.get("recharge_detected", False)
            )
            recharge_attempted = bool(
                context.get("recharge_authorized", False)
            )
            if recharge_succeeded:
                self.get_logger().info(
                    "Recharge interaction timed out after confirmed energy "
                    "transfer; recording a successful partial recharge."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    failure_reason=constants.FAILURE_NONE,
                    resume_exploration=False,
                    navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                    tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                    recharge_attempted=True,
                    recharge_succeeded=True,
                )
                return

            failure_reason = constants.FAILURE_RECHARGE_TIMEOUT
            event_description = (
                "energy_interaction_timeout_without_confirmed_transfer"
            )
        elif policy_id == self.POLICY_GOAL:
            failure_reason = constants.FAILURE_GOAL_NOT_CONFIRMED
            event_description = "goal_interaction_timeout_no_reward_detected"
            recharge_attempted = False
            recharge_succeeded = False
        else:
            failure_reason = constants.FAILURE_OBSERVATION_TIMEOUT
            event_description = "interaction_timeout"
            recharge_attempted = False
            recharge_succeeded = False

        self.get_logger().warn(event_description)
        self.finish_execution(
            generation,
            success=False,
            failure_reason=failure_reason,
            resume_exploration=self.resume_exploration_after_failure,
            navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
            tag_result=RobotinoPolicyOutcome.TAG_CHECK_INCONCLUSIVE,
            recharge_attempted=recharge_attempted,
            recharge_succeeded=recharge_succeeded,
        )

    def try_begin_energy_interaction_from_proximity(self) -> bool:
        """Switch from Nav2 motion to charging when the target is in range.

        The approach pose is intentionally conservative and may remain
        unreachable near a wall because of footprint inflation or final-yaw
        constraints.  Requiring NavigateToPose to report success before
        authorizing energy transfer can therefore leave Robotino visibly close
        to a usable bank without charging.  A matching, visible target inside
        the physical interaction radius is sufficient semantic arrival.
        """
        context = self.execution
        if context is None or context.get("stage") != self.STAGE_NAVIGATING:
            return False

        policy = context.get("policy")
        if policy is None:
            return False
        if int(policy.policy_id) != self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            return False

        state = self.latest_foraging_state
        if state is None:
            return False
        if not bool(getattr(state, "visible", False)):
            return False
        if int(getattr(state, "tag_id", -1)) != int(policy.target_tag_id):
            return False
        if not bool(getattr(state, "is_energy_bank", False)):
            return False

        try:
            distance = float(getattr(state, "distance", math.inf))
            remaining = float(getattr(state, "resource_remaining", 0.0))
        except (TypeError, ValueError):
            return False

        if not math.isfinite(distance) or not math.isfinite(remaining):
            return False
        if distance <= 0.05 or distance > self.energy_interaction_distance_m:
            return False
        if remaining <= self.resource_empty_epsilon:
            return False

        generation = int(context["generation"])
        current_energy = self.get_current_energy()

        # Change stage before canceling.  Generation/stage checks then cause
        # the asynchronous canceled Nav2 result to be treated as stale rather
        # than as a failed recovery policy.
        context["stage"] = self.STAGE_WAITING_INTERACTION
        context["arrival_energy"] = current_energy
        context["last_progress_energy"] = current_energy
        context["recharge_detected"] = False
        context["recharge_authorized"] = True
        context["interaction_deadline_ns"] = (
            self.get_clock().now().nanoseconds
            + int(self.interaction_timeout_s * 1e9)
        )
        context.pop("navigation_deadline_ns", None)

        goal_handle = self.active_navigation_goal_handle
        self.active_navigation_goal_handle = None
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    "Failed to cancel Nav2 after semantic arrival: "
                    f"{ex}"
                )

        self.stop_robot()
        self.set_recharge_target(int(policy.target_tag_id))
        self.get_logger().info(
            "Semantic energy arrival detected before exact Nav2 completion; "
            "canceling approach motion and authorizing transfer: "
            f"generation={generation}, "
            f"tag_id={int(policy.target_tag_id)}, "
            f"distance={distance:.3f}m, "
            f"interaction_radius={self.energy_interaction_distance_m:.3f}m, "
            f"remaining={remaining:.3f}, "
            f"energy={current_energy:.3f}."
        )
        return True

    def check_navigation_timeout(self) -> None:
        """Cancel a Nav2 goal that outlives the executor-owned deadline."""
        context = self.execution
        if context is None or context.get("stage") != self.STAGE_NAVIGATING:
            return

        deadline_ns = context.get("navigation_deadline_ns")
        if deadline_ns is None:
            # Compatibility for executions created before this field existed.
            context["navigation_deadline_ns"] = (
                self.get_clock().now().nanoseconds
                + int(self.navigation_timeout_s * 1e9)
            )
            return

        if self.get_clock().now().nanoseconds < int(deadline_ns):
            return

        generation = int(context["generation"])
        navigation_mode = str(context.get("navigation_mode", "tag"))
        resume_after_failure = (
            False
            if navigation_mode == "wander"
            else self.resume_exploration_after_failure
        )

        goal_handle = self.active_navigation_goal_handle
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    f"Failed to cancel timed-out Nav2 goal: {ex}"
                )

        self.stop_robot()
        self.get_logger().warn(
            "Nav2 execution timed out after "
            f"{self.navigation_timeout_s:.1f} s; "
            f"generation={generation}, mode={navigation_mode}."
        )
        self.finish_execution(
            generation,
            success=False,
            failure_reason=constants.FAILURE_NAVIGATION_TIMEOUT,
            resume_exploration=resume_after_failure,
            navigation_result=RobotinoPolicyOutcome.NAV_CANCELED,
        )

    # ------------------------------------------------------------------
    # Recovery-aware semantic completion
    # ------------------------------------------------------------------

    def check_interaction_completion(self) -> None:
        """Authorize charging after arrival, then report its real result."""
        context = self.get_waiting_context()
        if context is None:
            return

        policy = context["policy"]
        policy_id = int(policy.policy_id)
        generation = int(context["generation"])

        if self.latest_foraging_state is None:
            return

        if policy_id != self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            # Preserve the existing goal/reward behavior from the mixin.
            InteractionWaitingMixin.check_interaction_completion(self)
            return

        current_energy = self.get_current_energy()
        if context.get("arrival_energy") is None:
            context["arrival_energy"] = current_energy
            context["last_progress_energy"] = current_energy

        # Phase 1: Nav2 has succeeded, but collection is still disabled. Wait
        # until perception confirms the selected tag is visible and has energy.
        if not bool(context.get("recharge_authorized", False)):
            remaining = self.target_resource_remaining(policy)
            if remaining is None:
                return

            if remaining <= self.resource_empty_epsilon:
                self.get_logger().warn(
                    "Reached and found the selected energy bank, but it has "
                    "no usable resource: "
                    f"tag_id={int(policy.target_tag_id)}, "
                    f"remaining={remaining:.4f}."
                )
                self.finish_execution(
                    generation,
                    success=False,
                    failure_reason=constants.FAILURE_RECHARGE_FAILED,
                    resume_exploration=self.resume_exploration_after_failure,
                    navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                    tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                    recharge_attempted=True,
                    recharge_succeeded=False,
                )
                return

            context["recharge_authorized"] = True
            context["arrival_energy"] = current_energy
            context["last_progress_energy"] = current_energy
            context["recharge_detected"] = False
            context["interaction_deadline_ns"] = (
                self.get_clock().now().nanoseconds
                + int(self.interaction_timeout_s * 1e9)
            )
            self.set_recharge_target(int(policy.target_tag_id))
            self.get_logger().info(
                "Return-to-energy navigation and resource verification "
                "succeeded; authorizing transfer now: "
                f"tag_id={int(policy.target_tag_id)}, "
                f"remaining={remaining:.4f}, "
                f"energy={current_energy:.3f}."
            )
            return

        # Phase 2: only now may the memory/world model transfer resource.
        arrival_energy = float(context["arrival_energy"])
        energy_delta = current_energy - arrival_energy

        last_progress_energy = float(
            context.get("last_progress_energy", arrival_energy)
        )
        if current_energy - last_progress_energy >= self.energy_progress_delta:
            context["last_progress_energy"] = current_energy
            context["interaction_deadline_ns"] = (
                self.get_clock().now().nanoseconds
                + int(self.interaction_timeout_s * 1e9)
            )

        if energy_delta >= self.energy_success_delta:
            if not bool(context.get("recharge_detected", False)):
                context["recharge_detected"] = True
                self.get_logger().info(
                    "Authorized recharge confirmed: "
                    f"energy increased by {energy_delta:.3f}."
                )

        recharge_detected = bool(context.get("recharge_detected", False))
        if recharge_detected and current_energy >= self.resume_energy_threshold:
            self.get_logger().info(
                "Energy recovery complete: "
                f"energy={current_energy:.3f}, "
                f"threshold={self.resume_energy_threshold:.3f}, "
                f"gain={energy_delta:.3f}."
            )
            self.finish_execution(
                generation,
                success=True,
                failure_reason=constants.FAILURE_NONE,
                resume_exploration=self.resume_exploration_after_energy,
                navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                recharge_attempted=True,
                recharge_succeeded=True,
            )
            return

        remaining = self.target_resource_remaining(policy)
        if remaining is None or remaining > self.resource_empty_epsilon:
            return

        if recharge_detected:
            self.get_logger().info(
                "Energy bank drained for this visit after a successful "
                "partial recharge: "
                f"tag_id={int(policy.target_tag_id)}, "
                f"energy={current_energy:.3f}, gain={energy_delta:.3f}. "
                "The policy succeeds, but exploration remains disabled so "
                "eMDB can select another recovery action if still needed."
            )
            self.finish_execution(
                generation,
                success=True,
                failure_reason=constants.FAILURE_NONE,
                resume_exploration=False,
                navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                recharge_attempted=True,
                recharge_succeeded=True,
            )
            return

        self.get_logger().warn(
            "Authorized bank became empty without a confirmed energy increase: "
            f"tag_id={int(policy.target_tag_id)}, "
            f"remaining={remaining:.4f}."
        )
        self.finish_execution(
            generation,
            success=False,
            failure_reason=constants.FAILURE_RECHARGE_FAILED,
            resume_exploration=self.resume_exploration_after_failure,
            navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
            tag_result=RobotinoPolicyOutcome.TAG_FOUND,
            recharge_attempted=True,
            recharge_succeeded=False,
        )

    def target_resource_remaining(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> Optional[float]:
        """Return remaining resource only for a visible matching target tag."""
        state = self.latest_foraging_state
        if state is None or not hasattr(state, "resource_remaining"):
            return None

        visible = bool(
            getattr(
                state,
                "visible",
                getattr(state, "tag_visible", False),
            )
        )
        if not visible:
            return None

        observed_tag_id = int(getattr(state, "tag_id", -1))
        if observed_tag_id != int(policy.target_tag_id):
            return None

        if hasattr(state, "is_energy_bank") and not bool(state.is_energy_bank):
            return None

        try:
            return max(0.0, float(state.resource_remaining))
        except (TypeError, ValueError):
            return None


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
        node.set_wandering_active(False, force=True)
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()