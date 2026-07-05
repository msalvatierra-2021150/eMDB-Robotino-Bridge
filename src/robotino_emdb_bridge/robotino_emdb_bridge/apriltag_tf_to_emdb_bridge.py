import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from apriltag_msgs.msg import AprilTagDetectionArray

from tf2_ros import Buffer, TransformListener

from robotino_emdb_interfaces.msg import RobotinoTag


class AprilTagTFToEMDBBridge(Node):
    def __init__(self):
        super().__init__("apriltag_tf_to_emdb_bridge")

        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("output_topic", "/robotino/emdb/tag_detection")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_optical_frame")

        # -1 means use the strongest detected tag.
        # 3 means prioritize tag_3.
        self.declare_parameter("target_tag_id", -1)

        self.declare_parameter("confidence_margin_max", 120.0)
        self.declare_parameter("publish_empty_detection", True)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.map_frame = self.get_parameter("map_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.target_tag_id = int(self.get_parameter("target_tag_id").value)
        self.confidence_margin_max = float(
            self.get_parameter("confidence_margin_max").value
        )
        self.publish_empty_detection = bool(
            self.get_parameter("publish_empty_detection").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            AprilTagDetectionArray,
            self.detections_topic,
            self.detections_callback,
            10,
        )

        self.publisher = self.create_publisher(
            RobotinoTag,
            self.output_topic,
            10,
        )

        self.get_logger().info("AprilTag TF to e-MDB bridge started")
        self.get_logger().info(f"Listening to detections: {self.detections_topic}")
        self.get_logger().info(f"Publishing e-MDB tag data: {self.output_topic}")
        self.get_logger().info(f"Using map frame: {self.map_frame}")
        self.get_logger().info(f"Using robot frame: {self.robot_frame}")
        self.get_logger().info(f"Using camera frame: {self.camera_frame}")

    def detections_callback(self, msg: AprilTagDetectionArray):
        detections = list(msg.detections)

        if len(detections) == 0:
            if self.publish_empty_detection:
                self.publish_empty()
            return

        detection = self.select_detection(detections)

        if detection is None:
            if self.publish_empty_detection:
                self.publish_empty()
            return

        tag_id = int(detection.id)
        tag_frame = f"tag_{tag_id}"

        output = RobotinoTag()
        output.header = msg.header
        output.visible = True
        output.tag_id = tag_id
        output.family = str(detection.family)

        output.confidence = self.clamp(
            float(detection.decision_margin) / self.confidence_margin_max,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # 1. Camera → Tag
        # This gives distance and bearing.
        # ---------------------------------------------------------
        camera_to_tag = self.lookup_transform_safe(
            self.camera_frame,
            tag_frame,
        )

        if camera_to_tag is not None:
            tx = float(camera_to_tag.transform.translation.x)
            ty = float(camera_to_tag.transform.translation.y)
            tz = float(camera_to_tag.transform.translation.z)

            output.tag_x_camera = tx
            output.tag_y_camera = ty
            output.tag_z_camera = tz

            output.distance = math.sqrt(tx * tx + ty * ty + tz * tz)

            # In camera optical frame, z is usually forward and x is left/right.
            output.bearing = math.atan2(tx, tz)
        else:
            output.tag_x_camera = 0.0
            output.tag_y_camera = 0.0
            output.tag_z_camera = 0.0
            output.distance = 0.0
            output.bearing = 0.0

        # ---------------------------------------------------------
        # 2. Map → Tag
        # This is the semantic-map location of the tag.
        # ---------------------------------------------------------
        map_to_tag = self.lookup_transform_safe(
            self.map_frame,
            tag_frame,
        )

        if map_to_tag is not None:
            output.tag_x_map = float(map_to_tag.transform.translation.x)
            output.tag_y_map = float(map_to_tag.transform.translation.y)
            output.tag_yaw_map = self.yaw_from_quaternion(
                map_to_tag.transform.rotation
            )
        else:
            output.tag_x_map = 0.0
            output.tag_y_map = 0.0
            output.tag_yaw_map = 0.0

        # ---------------------------------------------------------
        # 3. Map → Robot
        # This is where the robot was when it observed the tag.
        # ---------------------------------------------------------
        map_to_robot = self.lookup_transform_safe(
            self.map_frame,
            self.robot_frame,
        )

        if map_to_robot is not None:
            output.robot_x_map = float(map_to_robot.transform.translation.x)
            output.robot_y_map = float(map_to_robot.transform.translation.y)
            output.robot_yaw_map = self.yaw_from_quaternion(
                map_to_robot.transform.rotation
            )
        else:
            output.robot_x_map = 0.0
            output.robot_y_map = 0.0
            output.robot_yaw_map = 0.0

        self.publisher.publish(output)

    def select_detection(self, detections):
        if self.target_tag_id >= 0:
            matching = [
                d for d in detections
                if int(d.id) == self.target_tag_id
            ]

            if len(matching) > 0:
                return max(
                    matching,
                    key=lambda d: float(d.decision_margin),
                )

            return None

        return max(
            detections,
            key=lambda d: float(d.decision_margin),
        )

    def lookup_transform_safe(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as error:
            self.get_logger().warn(
                f"Could not transform {target_frame} <- {source_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None

    def publish_empty(self):
        output = RobotinoTag()

        output.visible = False
        output.tag_id = -1
        output.family = ""

        output.confidence = 0.0

        output.distance = 0.0
        output.bearing = 0.0

        output.tag_x_camera = 0.0
        output.tag_y_camera = 0.0
        output.tag_z_camera = 0.0

        output.tag_x_map = 0.0
        output.tag_y_map = 0.0
        output.tag_yaw_map = 0.0

        output.robot_x_map = 0.0
        output.robot_y_map = 0.0
        output.robot_yaw_map = 0.0

        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.camera_frame

        self.publisher.publish(output)

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    def clamp(self, value, min_value, max_value):
        return float(max(min_value, min(max_value, value)))


def main(args=None):
    rclpy.init(args=args)

    node = AprilTagTFToEMDBBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()