import rclpy

from cognitive_nodes.perception import Perception
from core.utils import perception_dict_to_msg

from robotino_emdb_interfaces.msg import RobotinoForagingState


class RobotinoForagingPerception(Perception):
    def __init__(
        self,
        name="foraging_state",
        class_name="cognitive_nodes.perception.Perception",
        default_msg=None,
        default_topic=None,
        normalize_data=None,
        **params
    ):
        super().__init__(
            name=name,
            class_name=class_name,
            default_msg=default_msg,
            default_topic=default_topic,
            normalize_data=normalize_data,
            **params
        )

    def clamp(self, value, min_value=0.0, max_value=1.0):
        return max(min_value, min(max_value, float(value)))

    def normalize(self, value, min_value, max_value, default=0.0):
        if max_value == min_value:
            return default

        normalized = (float(value) - float(min_value)) / (
            float(max_value) - float(min_value)
        )

        return self.clamp(normalized)

    def process_and_send_reading(self):
        n = self.normalize_values

        p = {
            # Observation
            "valid": 1.0 if self.reading.valid else 0.0,
            "visible": 1.0 if self.reading.visible else 0.0,

            # One-hot tag identity
            "tag_0": 0.0,
            "tag_1": 0.0,
            "tag_2": 0.0,
            "tag_3": 0.0,
            "tag_4": 0.0,
            "tag_5": 0.0,

            # Memory
            "first_time_seen": 1.0 if self.reading.first_time_seen else 0.0,
            "known_tag": 1.0 if self.reading.known_tag else 0.0,
            "times_seen": self.normalize(
                self.reading.times_seen,
                0.0,
                n["times_seen_max"],
                default=0.0
            ),

            # Geometry
            "confidence": self.clamp(self.reading.confidence),
            "distance": self.normalize(
                self.reading.distance,
                n["distance_min"],
                n["distance_max"],
                default=0.0
            ),
            "bearing": self.normalize(
                self.reading.bearing,
                n["bearing_min"],
                n["bearing_max"],
                default=0.5
            ),
            "tag_x_map": self.normalize(
                self.reading.tag_x_map,
                n["map_x_min"],
                n["map_x_max"],
                default=0.0
            ),
            "tag_y_map": self.normalize(
                self.reading.tag_y_map,
                n["map_y_min"],
                n["map_y_max"],
                default=0.0
            ),
            "robot_x_map": self.normalize(
                self.reading.robot_x_map,
                n["map_x_min"],
                n["map_x_max"],
                default=0.0
            ),
            "robot_y_map": self.normalize(
                self.reading.robot_y_map,
                n["map_y_min"],
                n["map_y_max"],
                default=0.0
            ),

            # Foraging/resource meaning
            "is_energy_bank": 1.0 if self.reading.is_energy_bank else 0.0,
            "resource_available": 1.0 if self.reading.resource_available else 0.0,
            "resource_remaining": self.clamp(self.reading.resource_remaining),
            "resource_value": self.clamp(self.reading.resource_value),

            # Internal state
            "robot_energy": self.clamp(self.reading.robot_energy),
            "energy_need": self.clamp(self.reading.energy_need),

            # Rewards
            "novelty_reward": self.clamp(self.reading.novelty_reward),
            "energy_reward": self.clamp(self.reading.energy_reward),
            "goal_reward": self.clamp(self.reading.goal_reward),
            "total_reward": self.normalize(
                self.reading.total_reward,
                0.0,
                n["total_reward_max"],
                default=0.0
            ),

            # Remembered best bank
            "best_energy_bank_known": 1.0
            if self.reading.best_energy_tag_id >= 0
            else 0.0,
            "best_energy_x_map": self.normalize(
                self.reading.best_energy_x_map,
                n["map_x_min"],
                n["map_x_max"],
                default=0.0
            ),
            "best_energy_y_map": self.normalize(
                self.reading.best_energy_y_map,
                n["map_y_min"],
                n["map_y_max"],
                default=0.0
            ),
            "best_energy_score": self.clamp(self.reading.best_energy_score),

            # Goal
            "goal_satisfied": 1.0 if self.reading.goal_satisfied else 0.0,
        }

        tag_id = int(self.reading.tag_id)

        if 0 <= tag_id <= 5:
            p[f"tag_{tag_id}"] = 1.0

        sensor = {}
        sensor[self.name] = [p]

        sensor_msg = perception_dict_to_msg(sensor)

        self.publish_msg.perception = sensor_msg
        self.publish_msg.timestamp = self.get_clock().now().to_msg()

        self.perception_publisher.publish(self.publish_msg)


def main(args=None):
    rclpy.init(args=args)

    node = RobotinoForagingPerception(
        name="foraging_state",
        class_name="cognitive_nodes.perception.Perception",
        default_msg=RobotinoForagingState,
        default_topic="/robotino/emdb/foraging_state",
        normalize_data={
            "distance_min": 0.0,
            "distance_max": 3.0,
            "bearing_min": -1.57,
            "bearing_max": 1.57,
            "map_x_min": -3.0,
            "map_x_max": 3.0,
            "map_y_min": -3.0,
            "map_y_max": 3.0,
            "times_seen_max": 10.0,
            "total_reward_max": 3.0,
        }
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()