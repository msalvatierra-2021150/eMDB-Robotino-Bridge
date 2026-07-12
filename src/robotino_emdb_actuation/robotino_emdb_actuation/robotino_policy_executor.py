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

from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration

from robotino_emdb_actuation.navigation_geometry import (
    euclidean_distance,
    compute_approach_pose,
    yaw_to_quaternion_z_w,
)

ENERGY_APPROACH_STANDOFF_M = 0.5
GOAL_APPROACH_STANDOFF_M = 0.5

ENERGY_INTERACTION_DISTANCE_M = 0.55
GOAL_INTERACTION_DISTANCE_M = 0.55

MIN_DISTANCE_EPSILON_M = 0.001

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
    POLICY_GOAL = 4

    POLICY_TOPIC = "/robotino/emdb/selected_policy"
    FORAGING_TOPIC = "/robotino/emdb/foraging_state"
    OUTCOME_TOPIC = "/robotino/emdb/policy_outcome"

    CMD_VEL_TOPIC = "/cmd_vel"
    EXPLORATION_ENABLE_TOPIC = "/robotino/emdb/frontier_exploration_enable"

    NAV2_ACTION_NAME = "navigate_to_pose"
    MAP_FRAME = "map"

    MINIMUM_GOAL_INTERVAL_S = 5.0

    def __init__(self):
        super().__init__("robotino_policy_executor")
        
        self.active_goal_handle = None

        self.declare_parameter("robot_base_frame", "base_link")
        self.robot_base_frame = self.get_parameter("robot_base_frame").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Safety switch. Keep false until selected_policy looks correct.
        self.declare_parameter("enable_nav2_execution", False)

        self.policy_topic = self.POLICY_TOPIC
        self.foraging_topic = self.FORAGING_TOPIC
        self.outcome_topic = self.OUTCOME_TOPIC

        self.cmd_vel_topic = self.CMD_VEL_TOPIC
        self.exploration_enable_topic = self.EXPLORATION_ENABLE_TOPIC

        self.nav2_action_name = self.NAV2_ACTION_NAME
        self.map_frame = self.MAP_FRAME

        self.minimum_goal_interval = self.MINIMUM_GOAL_INTERVAL_S

        self.latest_foraging_state = None
        self.active_goal = False
        self.last_goal_time = None
        self.current_policy = None
        self.energy_before_policy = 0.0
        self.reward_before_policy = 0.0

        self.enable_nav2_execution = bool(
            self.get_parameter("enable_nav2_execution").value
        )

        self.exploration_enable_publisher = self.create_publisher(
            Bool,
            self.exploration_enable_topic,
            10,
        )

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

        self.nav2_client = ActionClient(
            self,
            NavigateToPose,
            self.nav2_action_name,
        )

    def get_robot_position_from_tf(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),  # latest available transform
                timeout=Duration(seconds=0.2),
            )

            x = tf_msg.transform.translation.x
            y = tf_msg.transform.translation.y

            return float(x), float(y)

        except TransformException as ex:
            self.get_logger().warn(
                f"Could not get current robot pose from TF "
                f"{self.map_frame} <- {self.robot_base_frame}: {ex}"
            )
            return None

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

        elif msg.policy_id == self.POLICY_GOAL:
            self.execute_goal(msg)

    def execute_continue_exploring(self, policy):
        self.set_exploration_enabled(True)
        self.cancel_active_navigation()
        self.publish_simple_outcome(policy, True, True, "continuing_exploration")

    def execute_search_for_energy(self, policy):
        self.set_exploration_enabled(True)
        self.cancel_active_navigation()
        self.publish_simple_outcome(policy, True, True, "searching_for_energy")

    def execute_inspect_visible_tag(self, policy):
        # First safe version: stop briefly.
        self.stop_robot()
        self.publish_simple_outcome(policy, True, True, "inspecting_visible_tag")

    def execute_goal(self, policy):
        self.nav_planning(policy, GOAL_INTERACTION_DISTANCE_M, GOAL_APPROACH_STANDOFF_M)       

    def execute_return_to_best_energy_bank(self, policy):
        self.nav_planning(policy, ENERGY_INTERACTION_DISTANCE_M, ENERGY_APPROACH_STANDOFF_M)

    def nav_planning(
        self,
        policy,
        interaction_distance,
        approach_standoff_distance,
    ):
        # Any policy-controlled Nav2 movement must stop frontier exploration.
        self.set_exploration_enabled(False)
        
        if not policy.use_nav2:
            self.publish_simple_outcome(
                policy,
                False,
                False,
                "policy_does_not_use_nav2",
            )
            return

        if not self.enable_nav2_execution:
            self.get_logger().warn(
                "Nav2 execution is disabled. "
                "Set enable_nav2_execution:=true to send goals."
            )
            self.publish_simple_outcome(
                policy,
                False,
                False,
                "dry_run_nav2_disabled",
            )
            return

        if self.active_goal:
            self.get_logger().debug(
                "A Nav2 goal is already active. Ignoring repeated policy."
            )
            return

        if not self.goal_interval_ok():
            self.get_logger().debug(
                "Ignoring policy because minimum goal interval has not elapsed."
            )
            return

        self.current_policy = policy
        self.energy_before_policy = self.get_current_energy()
        self.reward_before_policy = self.get_current_reward()

        target_x = float(policy.target_x_map)
        target_y = float(policy.target_y_map)

        # Navigate first to the robot pose from which the tag was last observed.
        goal_x = float(policy.last_seen_robot_x_map)
        goal_y = float(policy.last_seen_robot_y_map)

        # Option 1: preserve the orientation from the successful observation.
        goal_yaw = float(policy.last_seen_robot_yaw_map)

        robot_position = self.get_robot_position_from_tf()

        if robot_position is None:
            self.get_logger().warn(
                "No current robot TF pose available. Not sending Nav2 goal."
            )

            self.publish_simple_outcome(
                policy,
                False,
                False,
                "no_current_robot_tf_pose",
            )
            return

        robot_x, robot_y = robot_position

        distance_to_observation_pose = euclidean_distance(
            robot_x,
            robot_y,
            goal_x,
            goal_y,
        )

        self.get_logger().info(
            f"Distance to last successful observation pose: "
            f"{distance_to_observation_pose:.2f} m"
        )

        # Check arrival relative to the observation pose, not the tag itself.
        if distance_to_observation_pose <= interaction_distance:
            self.get_logger().info(
                "Already near the last successful tag-observation pose."
            )

            self.stop_robot()

            # Do not claim the energy bank itself was reached yet.
            # The camera must verify the expected tag.
            self.publish_simple_outcome(
                policy,
                True,
                True,
                "last_observation_pose_reached",
            )
            return

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.position.z = 0.0

        qz, qw = yaw_to_quaternion_z_w(goal_yaw)

        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f"Sending Nav2 goal to last successful observation pose for "
            f"policy {policy.policy_name}: "
            f"observation_x={goal_x:.2f}, "
            f"observation_y={goal_y:.2f}, "
            f"observation_yaw={goal_yaw:.2f}, "
            f"tag_x={target_x:.2f}, "
            f"tag_y={target_y:.2f}"
        )

        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 NavigateToPose action server not available"
            )

            self.publish_simple_outcome(
                policy,
                False,
                False,
                "nav2_server_unavailable",
            )
            return

        self.active_goal = True
        self.last_goal_time = self.get_clock().now()

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            self.nav2_goal_response_callback
        )

    def nav2_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as ex:
            self.get_logger().error(f"Failed to send Nav2 goal: {ex}")
            self.active_goal = False
            self.current_policy = None
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            self.active_goal = False

            if self.current_policy is not None:
                self.publish_simple_outcome(
                    self.current_policy,
                    False,
                    False,
                    "nav2_goal_rejected",
                )

            self.current_policy = None
            return

        self.active_goal_handle = goal_handle

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
    
    def cancel_active_navigation(self):
        if self.active_goal_handle is None:
            return

        self.active_goal_handle.cancel_goal_async()
        self.active_goal_handle = None
        self.active_goal = False

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