#!/usr/bin/env python3
"""Blocking bridge between official GII e-MDB policies and Robotino actuation.

The official GII ``PolicyBlocking`` node calls the
``cognitive_node_interfaces/srv/Policy`` service exposed here. This bridge:

1. receives the abstract e-MDB policy name;
2. builds the existing ``RobotinoSelectedPolicy`` command;
3. publishes it to the existing Robotino policy executor;
4. waits for the matching completed ``RobotinoPolicyOutcome``;
5. returns the objective success flag to the e-MDB policy service client.

The bridge deliberately does not calculate drives, rewards, or policy
activation. Those belong to the official e-MDB cognitive process.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from cognitive_node_interfaces.srv import Policy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)


@dataclass
class PendingExecution:
    """Description of the single policy execution currently being awaited."""

    requested_name: str
    expected_outcome_policy_id: int
    expected_target_id: Optional[int]
    event: threading.Event
    outcome: Optional[RobotinoPolicyOutcome] = None


class RobotinoPolicyExecutionBridge(Node):
    """Expose Robotino's existing policy executor as a GII blocking service."""

    # Existing RobotinoSelectedPolicy IDs used by robotino_policy_executor.
    POLICY_CONTINUE_EXPLORING = 0
    POLICY_INSPECT_VISIBLE_TAG = 1
    POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
    POLICY_SEARCH_FOR_ENERGY = 3
    POLICY_GOAL = 4

    def __init__(self) -> None:
        super().__init__("robotino_policy_execution_bridge")

        self.declare_parameter(
            "service_name",
            "/robotino/emdb/execute_policy",
        )
        self.declare_parameter(
            "foraging_topic",
            "/robotino/emdb/foraging_state",
        )
        self.declare_parameter(
            "selected_policy_topic",
            "/robotino/emdb/selected_policy",
        )
        self.declare_parameter(
            "outcome_topic",
            "/robotino/emdb/policy_outcome",
        )
        self.declare_parameter("execution_timeout_s", 120.0)
        self.declare_parameter("minimum_energy_bank_score", 0.0)
        self.declare_parameter("minimum_energy_bank_worthiness", 0.10)
        self.declare_parameter("wait_safe_duration_s", 1.0)
        self.declare_parameter("minimum_exploration_cycle_s", 1.0)

        self.service_name = str(self.get_parameter("service_name").value)
        self.foraging_topic = str(self.get_parameter("foraging_topic").value)
        self.selected_policy_topic = str(
            self.get_parameter("selected_policy_topic").value
        )
        self.outcome_topic = str(self.get_parameter("outcome_topic").value)
        self.execution_timeout_s = max(
            1.0,
            float(self.get_parameter("execution_timeout_s").value),
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
        self.wait_safe_duration_s = max(
            0.0,
            float(self.get_parameter("wait_safe_duration_s").value),
        )
        self.minimum_exploration_cycle_s = max(
            0.0,
            float(
                self.get_parameter(
                    "minimum_exploration_cycle_s"
                ).value
            ),
        )

        self.callback_group = ReentrantCallbackGroup()

        self.latest_foraging_state: Optional[RobotinoForagingState] = None
        self.pending: Optional[PendingExecution] = None
        self.pending_lock = threading.Lock()

        self.selected_policy_publisher = self.create_publisher(
            RobotinoSelectedPolicy,
            self.selected_policy_topic,
            10,
        )
        self.foraging_subscription = self.create_subscription(
            RobotinoForagingState,
            self.foraging_topic,
            self.foraging_callback,
            10,
            callback_group=self.callback_group,
        )
        self.outcome_subscription = self.create_subscription(
            RobotinoPolicyOutcome,
            self.outcome_topic,
            self.outcome_callback,
            10,
            callback_group=self.callback_group,
        )
        self.policy_service = self.create_service(
            Policy,
            self.service_name,
            self.execute_policy_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            "Robotino GII policy bridge ready: "
            f"service={self.service_name}, "
            f"command_topic={self.selected_policy_topic}, "
            f"outcome_topic={self.outcome_topic}"
        )

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        """Store the latest factual Robotino state used to resolve targets."""
        self.latest_foraging_state = msg

    def execute_policy_callback(
        self,
        request: Policy.Request,
        response: Policy.Response,
    ) -> Policy.Response:
        """Translate one official e-MDB policy request and wait for its outcome."""
        policy_name = self.canonical_policy_name(request.policy)

        if policy_name == "wait_safe":
            self.get_logger().info(
                "Executing e-MDB wait_safe fallback for "
                f"{self.wait_safe_duration_s:.2f} s"
            )
            if self.wait_safe_duration_s > 0.0:
                time.sleep(self.wait_safe_duration_s)
            response.success = True
            return response

        with self.pending_lock:
            if self.pending is not None:
                self.get_logger().warn(
                    "Rejected e-MDB policy request because another execution "
                    f"is active: {self.pending.requested_name}"
                )
                response.success = False
                return response

            command = self.build_selected_policy(policy_name)
            if command is None:
                response.success = False
                return response

            expected_outcome_policy_id = self.expected_outcome_policy_id(
                command.policy_id,
                state=self.latest_foraging_state,
            )
            expected_target_id = (
                int(command.target_tag_id)
                if int(command.target_tag_id) >= 0
                else None
            )

            pending = PendingExecution(
                requested_name=policy_name,
                expected_outcome_policy_id=expected_outcome_policy_id,
                expected_target_id=expected_target_id,
                event=threading.Event(),
            )
            self.pending = pending

        dispatch_started = time.monotonic()
        self.get_logger().info(
            "Dispatching official e-MDB policy "
            f"'{policy_name}' as Robotino policy_id={command.policy_id}, "
            f"target_id={command.target_tag_id}"
        )
        self.selected_policy_publisher.publish(command)

        completed = pending.event.wait(timeout=self.execution_timeout_s)

        with self.pending_lock:
            outcome = pending.outcome
            if self.pending is pending:
                self.pending = None

        if not completed or outcome is None:
            self.get_logger().error(
                f"Timed out waiting for completed outcome of '{policy_name}' "
                f"after {self.execution_timeout_s:.1f} s."
            )
            response.success = False
            return response

        response.success = bool(outcome.policy_success)

        # The current frontier executor acknowledges exploration immediately
        # while a separate frontier process keeps moving the robot. Without a
        # minimum policy window, MainLoop can consume every configured
        # iteration before energy or mapping state changes. Keep e-MDB in
        # control by allowing one fresh decision at a bounded cadence.
        if (
            response.success
            and policy_name in {
                "continue_exploring",
                "search_for_energy",
            }
            and self.minimum_exploration_cycle_s > 0.0
        ):
            elapsed = time.monotonic() - dispatch_started
            remaining = self.minimum_exploration_cycle_s - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

        self.get_logger().info(
            f"Official e-MDB policy '{policy_name}' completed: "
            f"success={response.success}, "
            f"failure_reason='{outcome.failure_reason}'"
        )
        return response

    def outcome_callback(self, msg: RobotinoPolicyOutcome) -> None:
        """Release the blocking service only for the matching final outcome."""
        if not bool(msg.policy_completed):
            return

        with self.pending_lock:
            pending = self.pending
            if pending is None:
                return

            if int(msg.policy_id) != int(pending.expected_outcome_policy_id):
                return

            if (
                pending.expected_target_id is not None
                and int(msg.target_id) != int(pending.expected_target_id)
            ):
                return

            pending.outcome = msg
            pending.event.set()

    def build_selected_policy(
        self,
        policy_name: str,
    ) -> Optional[RobotinoSelectedPolicy]:
        """Create the command consumed by the existing Robotino executor."""
        state = self.latest_foraging_state
        if state is None:
            self.get_logger().error(
                "Cannot execute policy: no RobotinoForagingState received yet."
            )
            return None

        command = RobotinoSelectedPolicy()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = str(state.header.frame_id)
        command.valid = bool(state.valid)
        command.execute_now = True
        command.expected_utility = 0.0
        command.priority_confidence = 1.0
        command.target_tag_id = -1
        command.target_x_map = 0.0
        command.target_y_map = 0.0
        command.target_yaw_map = 0.0
        command.last_seen_robot_x_map = 0.0
        command.last_seen_robot_y_map = 0.0
        command.last_seen_robot_yaw_map = 0.0

        if not command.valid:
            self.get_logger().warn(
                f"Cannot execute '{policy_name}': foraging state is invalid."
            )
            return None

        if policy_name == "continue_exploring":
            command.policy_id = self.POLICY_CONTINUE_EXPLORING
            command.policy_name = "continue_exploring"
            command.drive_id = 1
            command.drive_name = "novelty"
            command.goal_name = "increase_environment_knowledge"
            command.use_nav2 = False
            command.interrupt_exploration = False
            return command

        if policy_name == "search_for_energy":
            command.policy_id = self.POLICY_SEARCH_FOR_ENERGY
            command.policy_name = "search_for_energy"
            command.drive_id = 2
            command.drive_name = "energy"
            command.goal_name = "discover_energy_resource"
            command.use_nav2 = False
            command.interrupt_exploration = False
            return command

        if policy_name == "inspect_visible_tag":
            if not bool(state.visible) or int(state.tag_id) < 0:
                self.get_logger().warn(
                    "Cannot inspect a visible tag because no valid tag is visible."
                )
                return None

            command.policy_id = self.POLICY_INSPECT_VISIBLE_TAG
            command.policy_name = "inspect_visible_tag"
            command.drive_id = 1
            command.drive_name = "novelty"
            command.goal_name = "record_semantic_observation"
            command.target_tag_id = int(state.tag_id)
            command.target_x_map = float(state.tag_x_map)
            command.target_y_map = float(state.tag_y_map)
            command.target_yaw_map = float(state.tag_yaw_map)
            command.last_seen_robot_x_map = float(state.robot_x_map)
            command.last_seen_robot_y_map = float(state.robot_y_map)
            command.last_seen_robot_yaw_map = float(state.robot_yaw_map)
            command.use_nav2 = False
            command.interrupt_exploration = False
            return command

        if policy_name == "return_to_energy":
            best_tag_id = int(state.best_energy_tag_id)
            best_score = float(state.best_energy_score)
            best_worthiness = float(
                getattr(state, "best_energy_worthiness", 0.0)
            )

            if best_tag_id < 0:
                self.get_logger().warn(
                    "Cannot return to energy: no remembered energy bank."
                )
                return None

            if best_score < self.minimum_energy_bank_score:
                self.get_logger().warn(
                    "Cannot return to energy: best bank score "
                    f"{best_score:.3f} is below threshold "
                    f"{self.minimum_energy_bank_score:.3f}."
                )
                return None

            if best_worthiness < self.minimum_energy_bank_worthiness:
                self.get_logger().warn(
                    "Cannot return to energy: best bank worthiness "
                    f"{best_worthiness:.3f} is below threshold "
                    f"{self.minimum_energy_bank_worthiness:.3f}."
                )
                return None

            values = (
                float(state.best_energy_x_map),
                float(state.best_energy_y_map),
                float(state.best_energy_last_seen_robot_x_map),
                float(state.best_energy_last_seen_robot_y_map),
                float(state.best_energy_last_seen_robot_yaw_map),
            )
            if not all(math.isfinite(value) for value in values):
                self.get_logger().error(
                    f"Cannot return to energy: invalid remembered target {values}."
                )
                return None

            command.policy_id = self.POLICY_RETURN_TO_BEST_ENERGY_BANK
            command.policy_name = "return_to_best_energy_bank"
            command.drive_id = 2
            command.drive_name = "energy"
            command.goal_name = "recover_energy"
            command.target_tag_id = best_tag_id
            command.target_x_map = values[0]
            command.target_y_map = values[1]
            command.target_yaw_map = 0.0
            command.last_seen_robot_x_map = values[2]
            command.last_seen_robot_y_map = values[3]
            command.last_seen_robot_yaw_map = values[4]
            command.expected_utility = best_score
            command.use_nav2 = True
            command.interrupt_exploration = True
            return command

        if policy_name == "go_to_goal":
            return self.build_goal_policy(state, command)

        self.get_logger().error(
            f"Unsupported official e-MDB policy name: '{policy_name}'."
        )
        return None

    def build_goal_policy(
        self,
        state: RobotinoForagingState,
        command: RobotinoSelectedPolicy,
    ) -> Optional[RobotinoSelectedPolicy]:
        """Resolve a remembered goal when goal-memory fields are available."""
        goal_known = bool(getattr(state, "goal_known", False))
        goal_tag_id = int(getattr(state, "goal_tag_id", -1))

        # Temporary fallback: a currently visible goal can be executed even
        # before dedicated remembered-goal fields are added to the message.
        visible_goal = (
            bool(state.visible)
            and int(state.tag_id) >= 0
            and "goal" in str(state.tag_type).strip().lower()
        )

        if goal_known and goal_tag_id >= 0:
            target_x = float(getattr(state, "goal_x_map", 0.0))
            target_y = float(getattr(state, "goal_y_map", 0.0))
            observation_x = float(
                getattr(state, "goal_last_seen_robot_x_map", 0.0)
            )
            observation_y = float(
                getattr(state, "goal_last_seen_robot_y_map", 0.0)
            )
            observation_yaw = float(
                getattr(state, "goal_last_seen_robot_yaw_map", 0.0)
            )
        elif visible_goal:
            goal_tag_id = int(state.tag_id)
            target_x = float(state.tag_x_map)
            target_y = float(state.tag_y_map)
            observation_x = float(state.robot_x_map)
            observation_y = float(state.robot_y_map)
            observation_yaw = float(state.robot_yaw_map)
        else:
            self.get_logger().warn(
                "Cannot go to goal: no remembered goal target fields are "
                "available in RobotinoForagingState."
            )
            return None

        values = (
            target_x,
            target_y,
            observation_x,
            observation_y,
            observation_yaw,
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error(
                f"Cannot go to goal: invalid remembered goal target {values}."
            )
            return None

        command.policy_id = self.POLICY_GOAL
        command.policy_name = "go_to_goal"
        command.drive_id = 3
        command.drive_name = "goal"
        command.goal_name = "complete_task"
        command.target_tag_id = goal_tag_id
        command.target_x_map = target_x
        command.target_y_map = target_y
        command.target_yaw_map = 0.0
        command.last_seen_robot_x_map = observation_x
        command.last_seen_robot_y_map = observation_y
        command.last_seen_robot_yaw_map = observation_yaw
        command.use_nav2 = True
        command.interrupt_exploration = True
        return command

    @staticmethod
    def canonical_policy_name(raw_name: str) -> str:
        aliases = {
            "explore_frontier": "continue_exploring",
            "continue_exploring": "continue_exploring",
            "search_for_energy": "search_for_energy",
            "inspect_visible_tag": "inspect_visible_tag",
            "return_to_best_energy_bank": "return_to_energy",
            "return_to_energy": "return_to_energy",
            "goal_reached": "go_to_goal",
            "go_to_goal": "go_to_goal",
            "wait": "wait_safe",
            "wait_safe": "wait_safe",
        }
        normalized = str(raw_name).strip().lower()
        return aliases.get(normalized, normalized)

    @staticmethod
    def expected_outcome_policy_id(
        selected_policy_id: int,
        state: Optional[RobotinoForagingState] = None,
    ) -> int:
        if selected_policy_id in (
            RobotinoPolicyExecutionBridge.POLICY_CONTINUE_EXPLORING,
            RobotinoPolicyExecutionBridge.POLICY_SEARCH_FOR_ENERGY,
        ):
            return RobotinoPolicyOutcome.POLICY_EXPLORE
        if selected_policy_id == (
            RobotinoPolicyExecutionBridge.POLICY_INSPECT_VISIBLE_TAG
        ):
            if state is not None and bool(getattr(state, "is_energy_bank", False)):
                return RobotinoPolicyOutcome.POLICY_VERIFY_ENERGY
            return RobotinoPolicyOutcome.POLICY_UNKNOWN
        if selected_policy_id == (
            RobotinoPolicyExecutionBridge.POLICY_RETURN_TO_BEST_ENERGY_BANK
        ):
            return RobotinoPolicyOutcome.POLICY_RETURN_TO_ENERGY
        if selected_policy_id == RobotinoPolicyExecutionBridge.POLICY_GOAL:
            return RobotinoPolicyOutcome.POLICY_GO_TO_GOAL
        return RobotinoPolicyOutcome.POLICY_UNKNOWN


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotinoPolicyExecutionBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()