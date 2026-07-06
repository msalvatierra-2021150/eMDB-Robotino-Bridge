import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool

from robotino_emdb_interfaces.msg import (
    RobotinoSelectedPolicy,
    RobotinoPolicyOutcome,
    RobotinoForagingState,
)


class RobotinoPolicyExecutor(Node):
    """
    Actuation layer.

    Input:
        /robotino/emdb/selected_policy

    Optional input:
        /robotino/emdb/foraging_state

    Output:
        /robotino/emdb/policy_outcome

    Purpose:
        Execute selected policies using Nav2, frontier exploration control,
        or simple stop commands.
    """

    POLICY_CONTINUE_EXPLORING = 0
    POLICY_INSPECT_VISIBLE_TAG = 1
    POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
    POLICY_SEARCH_FOR_ENERGY = 3
    POLICY_GOAL_REACHED = 4

    def __init__(self):
        super().__init__("robotino_policy_executor")

        self.declare_parameter("policy_topic", "/robotino/emdb/selected_policy")
        self.declare_parameter("foraging_topic", "/robotino/emdb/foraging_state")
        self.declare_parameter("outcome_topic", "/robotino/emdb/policy_outcome")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("exploration_enable_topic", "/robotino/emdb/exploration_enable")

        self.declare_parameter("nav2_action_name", "navigate_to_pose")
        self.declare_parameter("map_frame", "map")

        # Safety switch. Keep false until selected_policy looks correct.
        self.declare_parameter("enable_nav2_execution", False)

        # If true, publishes Bool to exploration_enable_topic.
        # You can later connect this to your frontier exploration manager.
        self.declare_parameter("publish_exploration_control", False)

        self.declare_parameter("minimum_goal_interval", 5.0)

        self.policy_topic = self.get_parameter("policy_topic").value
        self.foraging_topic = self.get_parameter("foraging_topic").value
        self.outcome_topic = self.get_parameter("outcome_topic").value

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.exploration_enable_topic = self.get_parameter("exploration_enable_topic").value

        self.nav2_action_name = self.get_parameter("nav2_action_name").value
        self.map_frame = self.get_parameter("map_frame").value

        self.enable_nav2_execution = bool(
            self.get_parameter("enable_nav2_execution").value
        )
        self.publish_exploration_control = bool(
            self.get_parameter("publish_exploration_control").value
        )

        self.minimum_goal_interval = float(
            self.get_parameter("minimum_goal_interval").value
        )

        self.latest_foraging_state = None
        self.active_goal = False
        self.last_goal_time = None
        self.current_policy = None
        self.energy_before_policy = 0.0
        self.reward_before_policy = 0.0

        self.policy_subscriber = self.create_subscription(
            RobotinoSelectedPolicy,
            self.policy_topic,
            self.policy_callback,
            10,
        )

        self.foraging_subscriber = self.create_subscription(
            RobotinoForagingState,
            self.foraging_topic,
            self.foraging_callback,
            10,
        )

        self.outcome_publisher = self.create_publisher(
            RobotinoPolicyOutcome,
            self.outcome_topic,
            10,
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        self.exploration_enable_publisher = self.create_publisher(
            Bool,
            self.exploration_enable_topic,
            10,
        )

        self.nav2_client = ActionClient(
            self,
            NavigateToPose,
            self.nav2_action_name,
        )

        self.get_logger().info("Robotino policy executor started")
        self.get_logger().info(f"Subscribing to: {self.policy_topic}")
        self.get_logger().info(f"Publishing outcomes to: {self.outcome_topic}")
        self.get_logger().info(f"Nav2 execution enabled: {self.enable_nav2_execution}")

    def foraging_callback(self, msg: RobotinoForagingState):
        self.latest_foraging_state = msg

    def policy_callback(self, msg: RobotinoSelectedPolicy):
        if not msg.valid:
            return

        if not msg.execute_now:
            return

        if msg.policy_id == self.POLICY_CONTINUE_EXPLORING:
            self.execute_continue_exploring(msg)

        elif msg.policy_id == self.POLICY_SEARCH_FOR_ENERGY:
            self.execute_search_for_energy(msg)

        elif msg.policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            self.execute_inspect_visible_tag(msg)

        elif msg.policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.execute_return_to_best_energy_bank(msg)

        elif msg.policy_id == self.POLICY_GOAL_REACHED:
            self.execute_goal_reached(msg)

    def execute_continue_exploring(self, policy):
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(policy, True, True, "continuing_exploration")

    def execute_search_for_energy(self, policy):
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(policy, True, True, "searching_for_energy")

    def execute_inspect_visible_tag(self, policy):
        # First safe version: stop briefly.
        self.stop_robot()
        self.publish_simple_outcome(policy, True, True, "inspecting_visible_tag")

    def execute_goal_reached(self, policy):
        self.set_exploration_enabled(False)
        self.stop_robot()
        self.publish_simple_outcome(policy, True, True, "goal_reached_stopping_robot")

    def execute_return_to_best_energy_bank(self, policy):
        self.set_exploration_enabled(False)

        if not policy.use_nav2:
            self.publish_simple_outcome(policy, False, False, "policy_does_not_use_nav2")
            return

        if not self.enable_nav2_execution:
            self.get_logger().warn(
                "Nav2 execution is disabled. Set enable_nav2_execution:=true to send goals."
            )
            self.publish_simple_outcome(policy, False, False, "dry_run_nav2_disabled")
            return

        if self.active_goal:
            return

        if not self.goal_interval_ok():
            return

        self.current_policy = policy
        self.energy_before_policy = self.get_current_energy()
        self.reward_before_policy = self.get_current_reward()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(policy.target_x_map)
        goal_msg.pose.pose.position.y = float(policy.target_y_map)
        goal_msg.pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion_z_w(float(policy.target_yaw_map))
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f"Sending Nav2 goal for policy {policy.policy_name}: "
            f"x={policy.target_x_map:.2f}, y={policy.target_y_map:.2f}"
        )

        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 NavigateToPose action server not available")
            self.publish_simple_outcome(policy, False, False, "nav2_server_unavailable")
            return

        self.active_goal = True
        self.last_goal_time = self.get_clock().now()

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.nav2_goal_response_callback)

    def nav2_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            self.active_goal = False

            if self.current_policy is not None:
                self.publish_simple_outcome(
                    self.current_policy,
                    True,
                    False,
                    "nav2_goal_rejected",
                )

            return

        self.get_logger().info("Nav2 goal accepted")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        result = future.result()

        success = result.status == 4  # STATUS_SUCCEEDED in action_msgs

        status_text = "nav2_goal_succeeded" if success else f"nav2_goal_finished_status_{result.status}"

        self.get_logger().info(status_text)

        self.active_goal = False

        if self.current_policy is not None:
            self.publish_policy_outcome(
                self.current_policy,
                started=True,
                finished=True,
                success=success,
                status=status_text,
            )

        self.current_policy = None

    def publish_simple_outcome(self, policy, started, success, status):
        self.publish_policy_outcome(
            policy,
            started=started,
            finished=True,
            success=success,
            status=status,
        )

    def publish_policy_outcome(self, policy, started, finished, success, status):
        outcome = RobotinoPolicyOutcome()

        outcome.header.stamp = self.get_clock().now().to_msg()
        outcome.header.frame_id = self.map_frame

        outcome.valid = True

        outcome.policy_id = int(policy.policy_id)
        outcome.policy_name = str(policy.policy_name)

        outcome.started = bool(started)
        outcome.finished = bool(finished)
        outcome.success = bool(success)

        outcome.status = str(status)

        outcome.target_tag_id = int(policy.target_tag_id)

        energy_after = self.get_current_energy()
        reward_after = self.get_current_reward()

        outcome.energy_before = float(self.energy_before_policy)
        outcome.energy_after = float(energy_after)
        outcome.energy_delta = float(energy_after - self.energy_before_policy)

        outcome.reward_before = float(self.reward_before_policy)
        outcome.reward_after = float(reward_after)
        outcome.reward_delta = float(reward_after - self.reward_before_policy)

        self.outcome_publisher.publish(outcome)

    def get_current_energy(self):
        if self.latest_foraging_state is None:
            return 0.0

        return float(self.latest_foraging_state.robot_energy)

    def get_current_reward(self):
        if self.latest_foraging_state is None:
            return 0.0

        return float(self.latest_foraging_state.total_reward)

    def set_exploration_enabled(self, enabled):
        if not self.publish_exploration_control:
            return

        msg = Bool()
        msg.data = bool(enabled)
        self.exploration_enable_publisher.publish(msg)

    def stop_robot(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)

    def goal_interval_ok(self):
        if self.last_goal_time is None:
            return True

        elapsed = (
            self.get_clock().now().nanoseconds
            - self.last_goal_time.nanoseconds
        ) / 1e9

        return elapsed >= self.minimum_goal_interval

    def yaw_to_quaternion_z_w(self, yaw):
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def main(args=None):
    rclpy.init(args=args)

    node = RobotinoPolicyExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
