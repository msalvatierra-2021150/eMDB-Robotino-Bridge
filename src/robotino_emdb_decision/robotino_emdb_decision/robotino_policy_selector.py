import rclpy
from rclpy.node import Node

from robotino_emdb_interfaces.msg import (
    RobotinoMotivationState,
    RobotinoSelectedPolicy,
)

class RobotinoPolicySelector(Node):
    """
    Decision layer.

    Input:
        /robotino/emdb/motivation_state

    Output:
        /robotino/emdb/selected_policy

    Purpose:
        Convert dominant motivation into an explicit selected policy.
    """

    POLICY_CONTINUE_EXPLORING = 0
    POLICY_INSPECT_VISIBLE_TAG = 1
    POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
    POLICY_SEARCH_FOR_ENERGY = 3
    POLICY_GOAL_REACHED = 4

    def __init__(self):
        super().__init__("robotino_policy_selector")

        self.declare_parameter("input_topic", "/robotino/emdb/motivation_state")
        self.declare_parameter("output_topic", "/robotino/emdb/selected_policy")

        self.declare_parameter("minimum_confidence_to_execute", 0.0)
        self.declare_parameter("default_target_yaw", 0.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.minimum_confidence_to_execute = float(
            self.get_parameter("minimum_confidence_to_execute").value
        )
        self.default_target_yaw = float(
            self.get_parameter("default_target_yaw").value
        )
        self.subscriber = self.create_subscription(
            RobotinoMotivationState,
            self.input_topic,
            self.motivation_callback,
            10,
        )
        self.publisher = self.create_publisher(
            RobotinoSelectedPolicy,
            self.output_topic,
            10,
        )

        self.get_logger().info("Robotino policy selector started")
        self.get_logger().info(f"Subscribing to: {self.input_topic}")
        self.get_logger().info(f"Publishing to: {self.output_topic}")

    def motivation_callback(self, msg: RobotinoMotivationState):
        policy = RobotinoSelectedPolicy()

        policy.header = msg.header
        policy.valid = bool(msg.valid)

        policy.last_seen_robot_x_map = float(msg.last_seen_robot_x_map)
        policy.last_seen_robot_y_map = float(msg.last_seen_robot_y_map)
        policy.last_seen_robot_yaw_map = float(msg.last_seen_robot_yaw_map)

        policy.policy_id = int(msg.suggested_policy_id)
        policy.policy_name = str(msg.suggested_policy)

        policy.drive_id = int(msg.dominant_drive_id)
        policy.drive_name = str(msg.dominant_drive)

        policy.goal_name = str(msg.suggested_goal)

        policy.target_tag_id = int(msg.target_tag_id)
        policy.target_x_map = float(msg.target_x_map)
        policy.target_y_map = float(msg.target_y_map)
        policy.target_yaw_map = self.default_target_yaw

        policy.expected_utility = self.estimate_expected_utility(msg)
        policy.priority_confidence = float(msg.priority_confidence)

        policy.execute_now = self.should_execute(msg)
        policy.use_nav2 = self.should_use_nav2(msg)
        policy.interrupt_exploration = bool(msg.should_interrupt_exploration)

        self.publisher.publish(policy)

    def estimate_expected_utility(self, msg: RobotinoMotivationState):
        priorities = [
            float(msg.exploration_priority),
            float(msg.inspect_priority),
            float(msg.return_priority),
            float(msg.search_energy_priority),
            float(msg.goal_drive),
        ]

        return max(priorities)

    def should_execute(self, msg: RobotinoMotivationState):
        if not msg.valid:
            return False

        if msg.priority_confidence < self.minimum_confidence_to_execute:
            return False

        return True

    def should_use_nav2(self, msg: RobotinoMotivationState):
        return bool(
            msg.suggested_policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK
            and msg.should_send_nav_goal
            and msg.target_tag_id >= 0
        )


def main(args=None):
    rclpy.init(args=args)

    node = RobotinoPolicySelector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
