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

from typing import Any, Dict, List, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)

from .execution_lifecycle import ExecutionLifecycleMixin
from .geometry_helpers import GeometryHelpersMixin
from .interaction_waiting import InteractionWaitingMixin
from .navigation_execution import NavigationExecutionMixin
from .navigation_planning import NavigationPlanningMixin
from .outcome_publishing import OutcomePublishingMixin
from .policy_dispatch import PolicyDispatchMixin

MIN_DISTANCE_EPSILON_M = 0.001


class RobotinoPolicyExecutor(
    Node,
    PolicyDispatchMixin,
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