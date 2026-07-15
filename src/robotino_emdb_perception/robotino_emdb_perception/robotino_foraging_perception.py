#!/usr/bin/env python3
"""Robotino perception adapter for the official GII e-MDB architecture.

This class is created by the e-MDB Commander from the experiment YAML. It
subscribes to RobotinoForagingState, converts the relevant state to normalized
perceptions, and publishes the native e-MDB PerceptionStamped value topic.

The physical tag ID and map coordinates intentionally remain outside the
learned contextual vector. Robotino's resource memory and policy executor need
those concrete values, but P-Nodes should learn general contexts such as
"reliable reachable energy resource" rather than memorizing tag_2 or x=-1.7.
"""

import rclpy

from cognitive_nodes.perception import Perception
from core.utils import perception_dict_to_msg


class RobotinoForagingPerception(Perception):
    """Convert Robotino's foraging state into one normalized e-MDB sensor."""

    def __init__(
        self,
        name="foraging_state",
        class_name="cognitive_nodes.perception.Perception",
        default_msg=None,
        default_topic=None,
        normalize_data=None,
        **params,
    ):
        super().__init__(
            name=name,
            class_name="cognitive_nodes.perception.Perception",
            default_msg=default_msg,
            default_topic=default_topic,
            normalize_data=normalize_data or {},
            **params,
        )

    @staticmethod
    def clamp(value, min_value=0.0, max_value=1.0):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = min_value
        return max(min_value, min(max_value, numeric))

    def normalize(self, value, min_value, max_value, default=0.0):
        try:
            minimum = float(min_value)
            maximum = float(max_value)
            numeric = float(value)
        except (TypeError, ValueError):
            return self.clamp(default)

        if maximum <= minimum:
            return self.clamp(default)

        return self.clamp((numeric - minimum) / (maximum - minimum))

    @staticmethod
    def field(message, name, default):
        """Read a field while remaining compatible with the older ROS msg."""
        return getattr(message, name, default)

    def process_and_send_reading(self):
        """Publish the current Robotino state in native e-MDB format.

        Rewards are deliberately not included. The official e-MDB MainLoop
        obtains rewards from Goal/Drive nodes after policy execution and then
        creates episodes and updates P-Nodes/C-Nodes itself.
        """
        n = self.normalize_values or {}

        distance_min = n.get("distance_min", 0.0)
        distance_max = n.get("distance_max", 3.0)
        bearing_min = n.get("bearing_min", -3.141592653589793)
        bearing_max = n.get("bearing_max", 3.141592653589793)

        p = {
            # Current observation
            "observation_valid": 1.0 if self.reading.valid else 0.0,
            "tag_visible": 1.0 if self.reading.visible else 0.0,
            "first_time_seen": 1.0 if self.reading.first_time_seen else 0.0,
            "known_tag": 1.0 if self.reading.known_tag else 0.0,
            "detection_confidence": self.clamp(self.reading.confidence),
            "tag_distance": self.normalize(
                self.reading.distance,
                distance_min,
                distance_max,
                default=0.0,
            ),
            "tag_bearing": self.normalize(
                self.reading.bearing,
                bearing_min,
                bearing_max,
                default=0.5,
            ),

            # Meaning of the visible resource
            "is_energy_bank": 1.0 if self.reading.is_energy_bank else 0.0,
            "resource_available": (
                1.0 if self.reading.resource_available else 0.0
            ),
            "resource_remaining": self.clamp(
                self.reading.resource_remaining
            ),

            # Robot internal state
            "robot_energy": self.clamp(self.reading.robot_energy),
            "energy_need": self.clamp(self.reading.energy_need),

            # Best remembered candidate selected by Robotino resource memory
            "best_energy_bank_known": (
                1.0 if self.reading.best_energy_tag_id >= 0 else 0.0
            ),
            "best_energy_foraging_score": self.clamp(
                self.field(
                    self.reading,
                    "best_energy_foraging_score",
                    self.reading.best_energy_score,
                )
            ),
            "best_energy_presence": self.clamp(
                self.field(
                    self.reading,
                    "best_energy_presence_confidence",
                    0.5,
                )
            ),
            "best_energy_reachability": self.clamp(
                self.field(
                    self.reading,
                    "best_energy_reachability_confidence",
                    0.5,
                )
            ),
            "best_energy_recharge_reliability": self.clamp(
                self.field(
                    self.reading,
                    "best_energy_recharge_reliability",
                    0.5,
                )
            ),
            "best_energy_worthiness": self.clamp(
                self.field(
                    self.reading,
                    "best_energy_worthiness",
                    0.5,
                )
            ),
            "best_energy_score": self.clamp(
                self.reading.best_energy_score
            ),

            # Mission context. Seeing the goal and satisfying it are different.
            "goal_known": (
                1.0
                if self.field(self.reading, "goal_known", False)
                else 0.0
            ),
            "goal_satisfied": 1.0 if self.reading.goal_satisfied else 0.0,
        }

        sensor = {self.name: [p]}
        sensor_msg = perception_dict_to_msg(sensor)

        self.publish_msg.perception = sensor_msg
        self.publish_msg.timestamp = self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)

        self.get_logger().debug(
            f"Published native e-MDB perception {self.name}: {p}"
        )


def main(args=None):
    """Standalone test entry point.

    In the real integration the e-MDB Commander should instantiate this class
    from the experiment YAML instead of launching this executable manually.
    """
    rclpy.init(args=args)

    node = RobotinoForagingPerception(
        name="foraging_state",
        class_name="cognitive_nodes.perception.Perception",
        default_msg=(
            "robotino_emdb_interfaces.msg.RobotinoForagingState"
        ),
        default_topic="/robotino/emdb/foraging_state",
        normalize_data={
            "distance_min": 0.0,
            "distance_max": 3.0,
            "bearing_min": -3.141592653589793,
            "bearing_max": 3.141592653589793,
        },
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()