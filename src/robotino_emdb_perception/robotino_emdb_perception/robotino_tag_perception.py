from cognitive_nodes.perception import Perception
from core.utils import perception_dict_to_msg


class RobotinoTagPerception(Perception):
    def __init__(
        self,
        name="tag_detection",
        class_name="robotino_emdb_perception.robotino_tag_perception.RobotinoTagPerception",
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

        normalized = (float(value) - float(min_value)) / (float(max_value) - float(min_value))
        return self.clamp(normalized)

    def process_and_send_reading(self):
        sensor = {}

        # Default values
        perception = {
            "visible": 0.0,

            "tag_0": 0.0,
            "tag_1": 0.0,
            "tag_2": 0.0,
            "tag_3": 0.0,
            "tag_4": 0.0,
            "tag_5": 0.0,

            "confidence": 0.0,
            "distance": 0.0,
            "bearing": 0.5,

            "tag_x_map": 0.0,
            "tag_y_map": 0.0,

            "robot_x_map": 0.0,
            "robot_y_map": 0.0,
            "robot_yaw_map": 0.5,
        }

        if self.reading.visible:
            n = self.normalize_values

            tag_id = int(self.reading.tag_id)

            perception["visible"] = 1.0

            if 0 <= tag_id <= 5:
                perception[f"tag_{tag_id}"] = 1.0

            perception["confidence"] = self.clamp(self.reading.confidence)

            perception["distance"] = self.normalize(
                self.reading.distance,
                n["distance_min"],
                n["distance_max"],
                default=0.0
            )

            perception["bearing"] = self.normalize(
                self.reading.bearing,
                n["bearing_min"],
                n["bearing_max"],
                default=0.5
            )

            perception["tag_x_map"] = self.normalize(
                self.reading.tag_x_map,
                n["map_x_min"],
                n["map_x_max"],
                default=0.0
            )

            perception["tag_y_map"] = self.normalize(
                self.reading.tag_y_map,
                n["map_y_min"],
                n["map_y_max"],
                default=0.0
            )

            perception["robot_x_map"] = self.normalize(
                self.reading.robot_x_map,
                n["map_x_min"],
                n["map_x_max"],
                default=0.0
            )

            perception["robot_y_map"] = self.normalize(
                self.reading.robot_y_map,
                n["map_y_min"],
                n["map_y_max"],
                default=0.0
            )

            perception["robot_yaw_map"] = self.normalize(
                self.reading.robot_yaw_map,
                -3.1416,
                3.1416,
                default=0.5
            )

        sensor[self.name] = [perception]

        sensor_msg = perception_dict_to_msg(sensor)

        self.publish_msg.perception = sensor_msg
        self.publish_msg.timestamp = self.get_clock().now().to_msg()

        self.perception_publisher.publish(self.publish_msg)