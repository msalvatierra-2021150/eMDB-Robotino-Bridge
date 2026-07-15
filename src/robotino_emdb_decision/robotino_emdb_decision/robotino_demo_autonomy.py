#!/usr/bin/env python3
"""Minimal autonomous worthiness-foraging coordinator for a Robotino demo.

This node intentionally does not replace the official e-MDB MainLoop.  It is a
small deterministic bootstrap controller used while P-Node/C-Node policy
activation is still under development.  It closes the physical demo loop by
calling the same blocking policy service already used by official GII
PolicyBlocking nodes.

Decision order
--------------
1. Stop issuing commands when the mission is already satisfied.
2. Enter energy-recovery mode below ``low_energy_threshold`` and remain there
   until ``resume_energy_threshold`` is reached (hysteresis).
3. During recovery, return to the factual memory's best worthy energy bank.
4. If no worthy bank is known, keep frontier exploration enabled as
   ``search_for_energy``.
5. Outside recovery, go to a known goal after mapping completes.
6. Otherwise keep frontier exploration enabled as ``continue_exploring``.

The factual memory remains responsible for exact tag IDs, observation poses,
resource capacity, evidence, worthiness, and selecting the best bank.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from cognitive_node_interfaces.srv import Policy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from robotino_emdb_interfaces.msg import RobotinoForagingState


class RobotinoDemoAutonomy(Node):
    """Deterministic demo coordinator over the tested blocking policy bridge."""

    CONTINUOUS_POLICIES = {"continue_exploring", "search_for_energy"}

    def __init__(self) -> None:
        super().__init__("robotino_demo_autonomy")

        self.declare_parameter("enabled", True)
        self.declare_parameter(
            "foraging_topic", "/robotino/emdb/foraging_state"
        )
        self.declare_parameter(
            "mapping_complete_topic",
            "/frontier_exploration/mapping_complete",
        )
        self.declare_parameter(
            "policy_service", "/robotino/emdb/execute_policy"
        )
        self.declare_parameter("mode_topic", "/robotino/emdb/demo_mode")

        self.declare_parameter("decision_period_s", 0.5)
        self.declare_parameter("startup_delay_s", 2.0)
        self.declare_parameter("post_action_delay_s", 1.0)
        self.declare_parameter("recharge_settle_s", 4.0)
        self.declare_parameter("failed_return_cooldown_s", 15.0)
        self.declare_parameter("failed_goal_cooldown_s", 20.0)

        self.declare_parameter("low_energy_threshold", 0.35)
        self.declare_parameter("resume_energy_threshold", 0.70)
        self.declare_parameter("minimum_bank_worthiness", 0.10)
        self.declare_parameter("minimum_bank_score", 0.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.foraging_topic = str(
            self.get_parameter("foraging_topic").value
        )
        self.mapping_complete_topic = str(
            self.get_parameter("mapping_complete_topic").value
        )
        self.policy_service = str(
            self.get_parameter("policy_service").value
        )

        self.low_energy_threshold = self.clamp(
            self.get_parameter("low_energy_threshold").value
        )
        self.resume_energy_threshold = self.clamp(
            self.get_parameter("resume_energy_threshold").value
        )
        if self.resume_energy_threshold <= self.low_energy_threshold:
            self.get_logger().warn(
                "resume_energy_threshold must be above low_energy_threshold; "
                "using low + 0.10."
            )
            self.resume_energy_threshold = min(
                1.0, self.low_energy_threshold + 0.10
            )

        self.minimum_bank_worthiness = self.clamp(
            self.get_parameter("minimum_bank_worthiness").value
        )
        self.minimum_bank_score = max(
            0.0, float(self.get_parameter("minimum_bank_score").value)
        )
        self.startup_delay_s = max(
            0.0, float(self.get_parameter("startup_delay_s").value)
        )
        self.post_action_delay_s = max(
            0.0, float(self.get_parameter("post_action_delay_s").value)
        )
        self.recharge_settle_s = max(
            0.0, float(self.get_parameter("recharge_settle_s").value)
        )
        self.failed_return_cooldown_s = max(
            0.0,
            float(self.get_parameter("failed_return_cooldown_s").value),
        )
        self.failed_goal_cooldown_s = max(
            0.0,
            float(self.get_parameter("failed_goal_cooldown_s").value),
        )

        self.latest_state: Optional[RobotinoForagingState] = None
        self.mapping_complete = False
        self.energy_recovery_mode = False

        self.active_future = None
        self.active_policy: Optional[str] = None
        self.commanded_continuous_mode: Optional[str] = None

        now = self.now_seconds()
        self.start_after = now + self.startup_delay_s
        self.next_decision_after = self.start_after
        self.return_cooldown_until = 0.0
        self.goal_cooldown_until = 0.0
        self.last_status_log_time = 0.0
        self.last_published_mode: Optional[str] = None

        self.foraging_subscription = self.create_subscription(
            RobotinoForagingState,
            self.foraging_topic,
            self.foraging_callback,
            10,
        )
        self.mapping_subscription = None
        if self.mapping_complete_topic:
            self.mapping_subscription = self.create_subscription(
                Bool,
                self.mapping_complete_topic,
                self.mapping_complete_callback,
                10,
            )

        self.mode_publisher = self.create_publisher(
            String,
            str(self.get_parameter("mode_topic").value),
            10,
        )
        self.policy_client = self.create_client(Policy, self.policy_service)

        decision_period = max(
            0.1, float(self.get_parameter("decision_period_s").value)
        )
        self.decision_timer = self.create_timer(
            decision_period, self.decision_step
        )

        self.get_logger().info(
            "Robotino demo autonomy ready: "
            f"enabled={self.enabled}, low={self.low_energy_threshold:.2f}, "
            f"resume={self.resume_energy_threshold:.2f}, "
            f"minimum_worthiness={self.minimum_bank_worthiness:.2f}"
        )

    @staticmethod
    def clamp(value: object, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        if numeric != numeric:  # NaN
            numeric = default
        return max(0.0, min(1.0, numeric))

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        self.latest_state = msg

    def mapping_complete_callback(self, msg: Bool) -> None:
        self.mapping_complete = bool(msg.data)

    def publish_mode(self, mode: str) -> None:
        if mode == self.last_published_mode:
            return
        message = String()
        message.data = mode
        self.mode_publisher.publish(message)
        self.last_published_mode = mode
        self.get_logger().info(f"Autonomous demo mode -> {mode}")

    def bank_metrics(
        self, state: RobotinoForagingState
    ) -> tuple[int, float, float, bool]:
        tag_id = int(getattr(state, "best_energy_tag_id", -1))
        score = max(0.0, float(getattr(state, "best_energy_score", 0.0)))

        worthiness_field = getattr(state, "best_energy_worthiness", None)
        if worthiness_field is None:
            # Older message compatibility: the factual memory has already
            # folded worthiness into best_energy_score. Do not reject a bank
            # merely because the old message cannot publish the separated
            # probability yet.
            worthiness = 1.0 if tag_id >= 0 and score > 0.0 else 0.0
            worthy = tag_id >= 0 and score > self.minimum_bank_score
        else:
            worthiness = self.clamp(worthiness_field)
            worthy = (
                tag_id >= 0
                and score > self.minimum_bank_score
                and worthiness >= self.minimum_bank_worthiness
            )

        return tag_id, score, worthiness, worthy

    def goal_is_known(self, state: RobotinoForagingState) -> bool:
        if hasattr(state, "goal_known"):
            return bool(state.goal_known)
        return (
            bool(state.visible)
            and int(state.tag_id) >= 0
            and "goal" in str(state.tag_type).strip().lower()
        )

    def update_energy_mode(self, energy: float) -> None:
        previous = self.energy_recovery_mode
        if not self.energy_recovery_mode and energy <= self.low_energy_threshold:
            self.energy_recovery_mode = True
        elif self.energy_recovery_mode and energy >= self.resume_energy_threshold:
            self.energy_recovery_mode = False

        if previous != self.energy_recovery_mode:
            state = "entered" if self.energy_recovery_mode else "left"
            self.get_logger().info(
                f"{state.capitalize()} energy-recovery mode at "
                f"energy={energy:.3f}."
            )
            # A mode transition must be sent to the executor even when both
            # modes happen to use frontier exploration physically.
            self.commanded_continuous_mode = None

    def select_policy(
        self,
        state: RobotinoForagingState,
        now: float,
    ) -> tuple[Optional[str], str]:
        energy = self.clamp(state.robot_energy)
        self.update_energy_mode(energy)
        tag_id, score, worthiness, bank_worthy = self.bank_metrics(state)

        if bool(state.goal_satisfied):
            return None, "mission_complete"

        if self.energy_recovery_mode:
            if bank_worthy and now >= self.return_cooldown_until:
                return (
                    "return_to_energy",
                    f"low_energy_bank_{tag_id}_worthy_{worthiness:.2f}",
                )
            if bank_worthy:
                return (
                    "search_for_energy",
                    f"return_cooldown_bank_{tag_id}_score_{score:.2f}",
                )
            return (
                "search_for_energy",
                f"low_energy_no_worthy_bank_best_{worthiness:.2f}",
            )

        if (
            self.mapping_complete
            and self.goal_is_known(state)
            and now >= self.goal_cooldown_until
        ):
            return "go_to_goal", "mapping_complete_goal_known"

        if self.mapping_complete and self.goal_is_known(state):
            return None, "goal_retry_cooldown"

        if self.mapping_complete:
            return None, "mapping_complete_goal_unknown"

        return "continue_exploring", "energy_adequate_mapping_incomplete"

    def finish_active_request(self, now: float) -> None:
        if self.active_future is None or not self.active_future.done():
            return

        policy_name = self.active_policy or "unknown"
        success = False
        try:
            response = self.active_future.result()
            success = bool(response.success)
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(
                f"Policy service call '{policy_name}' failed: {error}"
            )

        self.get_logger().info(
            f"Autonomous policy '{policy_name}' completed: success={success}."
        )

        if policy_name == "return_to_energy":
            if success:
                self.next_decision_after = now + self.recharge_settle_s
            else:
                self.return_cooldown_until = (
                    now + self.failed_return_cooldown_s
                )
                self.next_decision_after = now + self.post_action_delay_s
                # Force one semantic search command after a failed bank.
                self.commanded_continuous_mode = None
        elif policy_name == "go_to_goal":
            if not success:
                self.goal_cooldown_until = now + self.failed_goal_cooldown_s
            self.next_decision_after = now + self.post_action_delay_s
        else:
            self.next_decision_after = now + self.post_action_delay_s

        self.active_future = None
        self.active_policy = None

    def dispatch_policy(self, policy_name: str, reason: str) -> None:
        request = Policy.Request()
        request.policy = policy_name
        self.active_policy = policy_name
        self.active_future = self.policy_client.call_async(request)

        if policy_name in self.CONTINUOUS_POLICIES:
            self.commanded_continuous_mode = policy_name

        state = self.latest_state
        energy = (
            self.clamp(state.robot_energy) if state is not None else 0.0
        )
        tag_id, score, worthiness, _ = (
            self.bank_metrics(state)
            if state is not None
            else (-1, 0.0, 0.0, False)
        )
        self.get_logger().info(
            f"Dispatching '{policy_name}': reason={reason}, "
            f"energy={energy:.3f}, best_bank={tag_id}, "
            f"score={score:.3f}, worthiness={worthiness:.3f}."
        )
        self.publish_mode(policy_name)

    def decision_step(self) -> None:
        if not self.enabled:
            self.publish_mode("disabled")
            return

        now = self.now_seconds()
        self.finish_active_request(now)
        if self.active_future is not None:
            return
        if now < self.next_decision_after:
            return

        state = self.latest_state
        if state is None or not bool(state.valid):
            if now - self.last_status_log_time >= 5.0:
                self.get_logger().warn(
                    "Waiting for a valid RobotinoForagingState."
                )
                self.last_status_log_time = now
            self.publish_mode("waiting_for_state")
            return

        if not self.policy_client.service_is_ready():
            self.policy_client.wait_for_service(timeout_sec=0.0)
            if now - self.last_status_log_time >= 5.0:
                self.get_logger().warn(
                    f"Waiting for policy service {self.policy_service}."
                )
                self.last_status_log_time = now
            self.publish_mode("waiting_for_policy_service")
            return

        desired_policy, reason = self.select_policy(state, now)
        if desired_policy is None:
            self.publish_mode(reason)
            return

        # Exploration and search are persistent physical modes. Sending the
        # same command every timer tick would create noisy, meaningless
        # outcomes, so only dispatch on a semantic mode transition.
        if (
            desired_policy in self.CONTINUOUS_POLICIES
            and desired_policy == self.commanded_continuous_mode
        ):
            self.publish_mode(desired_policy)
            return

        self.dispatch_policy(desired_policy, reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotinoDemoAutonomy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()