"""Publish Robotino factual state as official e-MDB drive inputs.

This node is only an adapter. It does not choose policies and it does not
calculate e-MDB rewards. The official DriveExponential and GoalMotiven nodes
consume these normalized satisfaction values and calculate drive evaluation
and reward themselves.

Published satisfaction semantics (all in [0, 1]):

* energy: 1 means the robot is fully charged.
* resource_knowledge: 1 means a highly trustworthy energy resource is known.
* exploration: 1 means mapping/exploration is complete.
* mission: 1 means the final mission goal has been completed.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from robotino_emdb_interfaces.msg import RobotinoForagingState


class RobotinoCognitiveSignals(Node):
    """Translate Robotino state into normalized e-MDB satisfaction topics."""

    def __init__(self) -> None:
        super().__init__("robotino_cognitive_signals")

        self.declare_parameter(
            "foraging_topic",
            "/robotino/emdb/foraging_state",
        )
        self.declare_parameter(
            "mapping_complete_topic",
            "/robotino/emdb/exploration_complete",
        )
        self.declare_parameter("mapping_progress_topic", "")

        self.declare_parameter(
            "energy_satisfaction_topic",
            "/robotino/emdb/satisfaction/energy",
        )
        self.declare_parameter(
            "resource_satisfaction_topic",
            "/robotino/emdb/satisfaction/resource_knowledge",
        )
        self.declare_parameter(
            "exploration_satisfaction_topic",
            "/robotino/emdb/satisfaction/exploration",
        )
        self.declare_parameter(
            "mission_satisfaction_topic",
            "/robotino/emdb/satisfaction/mission",
        )
        self.declare_parameter("publish_period_s", 0.2)

        self.foraging_topic = str(
            self.get_parameter("foraging_topic").value
        )
        self.mapping_complete_topic = str(
            self.get_parameter("mapping_complete_topic").value
        )
        self.mapping_progress_topic = str(
            self.get_parameter("mapping_progress_topic").value
        )

        self.latest_state: Optional[RobotinoForagingState] = None
        self.mapping_complete = False
        self.mapping_progress: Optional[float] = None

        self.energy_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("energy_satisfaction_topic").value),
            10,
        )
        self.resource_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("resource_satisfaction_topic").value),
            10,
        )
        self.exploration_publisher = self.create_publisher(
            Float32,
            str(
                self.get_parameter(
                    "exploration_satisfaction_topic"
                ).value
            ),
            10,
        )
        self.mission_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("mission_satisfaction_topic").value),
            10,
        )

        self.foraging_subscription = self.create_subscription(
            RobotinoForagingState,
            self.foraging_topic,
            self.foraging_callback,
            10,
        )

        self.mapping_complete_subscription = None
        if self.mapping_complete_topic:
            self.mapping_complete_subscription = self.create_subscription(
                Bool,
                self.mapping_complete_topic,
                self.mapping_complete_callback,
                10,
            )

        self.mapping_progress_subscription = None
        if self.mapping_progress_topic:
            self.mapping_progress_subscription = self.create_subscription(
                Float32,
                self.mapping_progress_topic,
                self.mapping_progress_callback,
                10,
            )

        period = max(
            0.05,
            float(self.get_parameter("publish_period_s").value),
        )
        self.publish_timer = self.create_timer(period, self.publish_signals)

        self.get_logger().info(
            "Robotino cognitive signal adapter ready: "
            f"foraging={self.foraging_topic}, "
            f"mapping_complete={self.mapping_complete_topic or '<disabled>'}, "
            f"mapping_progress={self.mapping_progress_topic or '<disabled>'}"
        )

    @staticmethod
    def clamp(value: object, default: float = 0.0) -> float:
        """Return a finite numeric value clamped to [0, 1]."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default

        if numeric != numeric:  # NaN
            numeric = default

        return max(0.0, min(1.0, numeric))

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        self.latest_state = msg

    def mapping_complete_callback(self, msg: Bool) -> None:
        self.mapping_complete = bool(msg.data)
        if self.mapping_complete:
            self.mapping_progress = 1.0

    def mapping_progress_callback(self, msg: Float32) -> None:
        self.mapping_progress = self.clamp(msg.data)
        if self.mapping_progress >= 1.0:
            self.mapping_complete = True

    def get_resource_satisfaction(
        self,
        state: RobotinoForagingState,
    ) -> float:
        """Return confidence that a useful energy resource is remembered."""
        if int(state.best_energy_tag_id) < 0:
            return 0.0

        # New message field from the adapted Robotino resource memory.
        worthiness = getattr(state, "best_energy_worthiness", None)
        if worthiness is not None:
            return self.clamp(worthiness)

        # Compatibility fallback for the older message definition.
        return self.clamp(getattr(state, "best_energy_score", 0.0))

    def publish_float(self, publisher, value: float) -> None:
        msg = Float32()
        msg.data = self.clamp(value)
        publisher.publish(msg)

    def publish_signals(self) -> None:
        """Publish one coherent satisfaction snapshot."""
        state = self.latest_state
        if state is None or not bool(state.valid):
            return

        energy = self.clamp(state.robot_energy)
        resource_knowledge = self.get_resource_satisfaction(state)
        exploration = (
            self.clamp(self.mapping_progress)
            if self.mapping_progress is not None
            else (1.0 if self.mapping_complete else 0.0)
        )
        mission = 1.0 if bool(state.goal_satisfied) else 0.0

        self.publish_float(self.energy_publisher, energy)
        self.publish_float(self.resource_publisher, resource_knowledge)
        self.publish_float(self.exploration_publisher, exploration)
        self.publish_float(self.mission_publisher, mission)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotinoCognitiveSignals()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
