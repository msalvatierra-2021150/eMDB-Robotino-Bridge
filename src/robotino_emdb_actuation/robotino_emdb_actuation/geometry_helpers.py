"""Geometry and ROS helpers.

Pure(ish) computations supporting navigation planning: reading the
robot's current pose from TF, choosing a safe approach axis toward a tag,
building PoseStamped goals, and measuring planned path length.
"""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.duration import Duration
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped

from robotino_emdb_interfaces.msg import RobotinoSelectedPolicy

MIN_DISTANCE_EPSILON_M = 0.001


class GeometryHelpersMixin:
    """Requires from the host class:

    Attributes: tf_buffer, map_frame, robot_base_frame, last_goal_time,
        minimum_goal_interval_s.
    """

    def get_robot_position_from_tf(self) -> Optional[Tuple[float, float]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            return (
                float(tf_msg.transform.translation.x),
                float(tf_msg.transform.translation.y),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"Could not get current robot pose from TF "
                f"{self.map_frame} <- {self.robot_base_frame}: {ex}"
            )
            return None

    def get_approach_axis(
        self,
        policy: RobotinoSelectedPolicy,
        target_x: float,
        target_y: float,
        observation_x: float,
        observation_y: float,
        robot_x: float,
        robot_y: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Return a unit vector from the tag toward known free space."""
        for x_field, y_field in (
            ("target_normal_x_map", "target_normal_y_map"),
            ("normal_x_map", "normal_y_map"),
        ):
            if not hasattr(policy, x_field) or not hasattr(policy, y_field):
                continue

            nx = float(getattr(policy, x_field))
            ny = float(getattr(policy, y_field))
            magnitude = math.hypot(nx, ny)

            if (
                math.isfinite(nx)
                and math.isfinite(ny)
                and magnitude > MIN_DISTANCE_EPSILON_M
            ):
                return (
                    nx / magnitude,
                    ny / magnitude,
                    f"{x_field}/{y_field}",
                )

        # The pose from which the tag was actually observed is a known-free
        # side and is safer than interpreting an arbitrary Euler yaw as the
        # wall-normal direction.
        dx = observation_x - target_x
        dy = observation_y - target_y
        magnitude = math.hypot(dx, dy)
        if magnitude > MIN_DISTANCE_EPSILON_M:
            return (
                dx / magnitude,
                dy / magnitude,
                "last_successful_observation_pose",
            )

        # Last-resort fallback if observation and target coordinates coincide.
        dx = robot_x - target_x
        dy = robot_y - target_y
        magnitude = math.hypot(dx, dy)
        if magnitude > MIN_DISTANCE_EPSILON_M:
            return (
                dx / magnitude,
                dy / magnitude,
                "current_robot_pose_fallback",
            )

        return None, None, "unavailable"

    def make_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        qz, qw = yaw_to_quaternion_z_w(yaw)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    @staticmethod
    def compute_path_length(path) -> float:
        if len(path.poses) < 2:
            return 0.0

        total = 0.0
        previous = path.poses[0].pose.position

        for pose_stamped in path.poses[1:]:
            current = pose_stamped.pose.position
            total += math.hypot(
                float(current.x) - float(previous.x),
                float(current.y) - float(previous.y),
            )
            previous = current

        return total

    def goal_interval_ok(self) -> bool:
        if self.last_goal_time is None:
            return True

        elapsed_s = (
            self.get_clock().now().nanoseconds
            - self.last_goal_time.nanoseconds
        ) / 1e9
        return elapsed_s >= self.minimum_goal_interval_s
    
def yaw_to_quaternion_z_w(yaw):
    return math.sin(float(yaw) / 2.0), math.cos(float(yaw) / 2.0)