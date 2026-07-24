#!/usr/bin/env python3
"""Compact Robotino context perception for official GII e-MDB selection.

This Perception does not select policies.  It publishes a normalized context
that seeded P-Nodes can recognize.  The official MainLoop then selects among
PolicyBlocking nodes through P-Node -> C-Node -> Policy activation.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from cognitive_nodes.perception import Perception
from core.utils import perception_dict_to_msg
from std_msgs.msg import Bool, Float32


class RobotinoContextPerception(Perception):
    """Publish discrete, hysteretic foraging contexts as an e-MDB sensor."""

    def __init__(
        self,
        name: str = "robotino_context",
        class_name: str = "cognitive_nodes.perception.Perception",
        default_msg: str = "robotino_emdb_interfaces.msg.RobotinoForagingState",
        default_topic: str = "/robotino/emdb/foraging_state",
        mapping_complete_topic: str = "",
        exploration_satisfaction_topic: str = (
            "/robotino/emdb/satisfaction/exploration"
        ),
        frontiers_available_topic: str = "",
        low_energy_threshold: float = 0.35,
        resume_energy_threshold: float = 0.70,
        minimum_bank_worthiness: float = 0.10,
        **params,
    ) -> None:
        # Force the canonical e-MDB type.  Commander still imports this custom
        # implementation from YAML, while LTM stores it under "Perception".
        super().__init__(
            name=name,
            class_name="cognitive_nodes.perception.Perception",
            default_msg=default_msg,
            default_topic=default_topic,
            normalize_data={},
            **params,
        )

        self.low_energy_threshold = self.clamp(low_energy_threshold)
        self.resume_energy_threshold = self.clamp(resume_energy_threshold)
        if self.resume_energy_threshold <= self.low_energy_threshold:
            self.resume_energy_threshold = min(
                1.0, self.low_energy_threshold + 0.10
            )
        self.minimum_bank_worthiness = self.clamp(
            minimum_bank_worthiness
        )

        self.mapping_complete = False
        self.frontiers_available = True
        self.frontiers_available_received = False
        self.energy_recovery_mode = False
        self.energy_mode_initialized = False

        self.mapping_complete_subscription = None
        if mapping_complete_topic:
            self.mapping_complete_subscription = self.create_subscription(
                Bool,
                mapping_complete_topic,
                self.mapping_complete_callback,
                10,
            )

        self.exploration_satisfaction_subscription = None
        if exploration_satisfaction_topic:
            self.exploration_satisfaction_subscription = (
                self.create_subscription(
                    Float32,
                    exploration_satisfaction_topic,
                    self.exploration_satisfaction_callback,
                    10,
                )
            )

        self.frontiers_available_subscription = None
        if frontiers_available_topic:
            self.frontiers_available_subscription = self.create_subscription(
                Bool,
                frontiers_available_topic,
                self.frontiers_available_callback,
                10,
            )

        self.get_logger().info(
            "Robotino context perception ready: "
            f"low={self.low_energy_threshold:.2f}, "
            f"resume={self.resume_energy_threshold:.2f}, "
            f"minimum_worthiness={self.minimum_bank_worthiness:.2f}, "
            f"mapping_complete_topic={mapping_complete_topic or '<disabled>'}, "
            f"exploration_satisfaction_topic="
            f"{exploration_satisfaction_topic or '<disabled>'}, "
            f"frontiers_available_topic="
            f"{frontiers_available_topic or '<infer from mapping>'}"
        )

    @staticmethod
    def clamp(value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        if number != number:  # NaN
            number = default
        return max(0.0, min(1.0, number))

    def mapping_complete_callback(self, msg: Bool) -> None:
        self.mapping_complete = bool(msg.data)
        if self.mapping_complete:
            self.frontiers_available = False

    def exploration_satisfaction_callback(self, msg: Float32) -> None:
        self.mapping_complete = self.clamp(msg.data) >= 0.999
        if self.mapping_complete:
            self.frontiers_available = False

    def frontiers_available_callback(self, msg: Bool) -> None:
        self.frontiers_available_received = True
        self.frontiers_available = bool(msg.data)

    def update_energy_mode(self, energy: float) -> None:
        if not self.energy_mode_initialized:
            self.energy_recovery_mode = (
                energy <= self.low_energy_threshold
            )
            self.energy_mode_initialized = True
            return

        if (
            not self.energy_recovery_mode
            and energy <= self.low_energy_threshold
        ):
            self.energy_recovery_mode = True
            self.get_logger().info(
                f"Entered energy-recovery context at energy={energy:.3f}"
            )
        elif (
            self.energy_recovery_mode
            and energy >= self.resume_energy_threshold
        ):
            self.energy_recovery_mode = False
            self.get_logger().info(
                f"Left energy-recovery context at energy={energy:.3f}"
            )

    def process_and_send_reading(self) -> None:
        state = self.reading
        energy = self.clamp(getattr(state, "robot_energy", 0.0))
        self.update_energy_mode(energy)

        best_tag_id = int(getattr(state, "best_energy_tag_id", -1))
        best_score = max(0.0, float(getattr(state, "best_energy_score", 0.0)))
        # has_worthiness_field = hasattr(state, "best_energy_worthiness")
        best_worthiness = self.clamp(
            getattr(state, "best_energy_worthiness", 0.0)
        )
        bank_known = best_tag_id >= 0
        # Backward compatibility: older RobotinoForagingState definitions did
        # not contain best_energy_worthiness. In that case the memory's
        # positive best_energy_score is already reliability-weighted, so a
        # known candidate is considered worthy rather than being forced false.
        bank_worthy = (
            best_tag_id >= 0
            and best_score > 0.0
        )

        goal_known = bool(getattr(state, "goal_known", False))
        if not goal_known:
            goal_known = bool(
                getattr(state, "visible", False)
                and "goal" in str(getattr(state, "tag_type", "")).lower()
            )
        goal_satisfied = bool(getattr(state, "goal_satisfied", False))

        if self.frontiers_available_received:
            frontiers_available = self.frontiers_available
        else:
            # Safe compatibility fallback for frontier implementations that
            # publish only mapping_complete.
            frontiers_available = not self.mapping_complete

        p = {
            "state_valid": 1.0 if bool(getattr(state, "valid", False)) else 0.0,
            "energy_recovery_mode": (
                1.0 if self.energy_recovery_mode else 0.0
            ),
            "energy_adequate": (
                0.0 if self.energy_recovery_mode else 1.0
            ),
            "bank_known": 1.0 if bank_known else 0.0,
            "bank_worthy": 1.0 if bank_worthy else 0.0,
            "bank_worthiness": best_worthiness,
            "mapping_complete": 1.0 if self.mapping_complete else 0.0,
            "frontiers_available": 1.0 if frontiers_available else 0.0,
            "goal_known": 1.0 if goal_known else 0.0,
            "goal_satisfied": 1.0 if goal_satisfied else 0.0,
        }

        self.publish_msg.perception = perception_dict_to_msg(
            {self.name: [p]}
        )
        self.publish_msg.timestamp = self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)
        self.get_logger().debug(f"Published Robotino context: {p}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotinoContextPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()