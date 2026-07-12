import rclpy
import math
from action_msgs.msg import GoalStatus
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
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
    yaw_to_quaternion_z_w,
    normalize_angle,
    quaternion_to_yaw

)

# Center-of-base distance from the tag/wall.
# Robotino radius is about 0.20 m, so 0.65 m leaves room for inflation
# and localization error. Reduce only after testing.
ENERGY_APPROACH_STANDOFF_M = 0.65
GOAL_APPROACH_STANDOFF_M = 0.65

ENERGY_INTERACTION_DISTANCE_M = 0.55
GOAL_INTERACTION_DISTANCE_M = 0.55

OBSERVATION_POSITION_TOLERANCE_M = 0.05
OBSERVATION_YAW_TOLERANCE_RAD = math.radians(10.0)

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
    COMPUTE_PATH_ACTION_NAME = "compute_path_to_pose"
    MAP_FRAME = "map"

    MINIMUM_GOAL_INTERVAL_S = 5.0

    def __init__(self):
        super().__init__("robotino_policy_executor")
        
        self.active_goal_handle = None
        self.active_path_goal_handle = None

        # Non-blocking candidate-planning state.
        self.planning_in_progress = False
        self.candidate_plan_generation = 0
        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0
        self.selected_candidate_name = None

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
        self.compute_path_action_name = self.COMPUTE_PATH_ACTION_NAME
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

        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            self.compute_path_action_name,
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
        
    def get_robot_pose_from_tf(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )

            x = float(tf_msg.transform.translation.x)
            y = float(tf_msg.transform.translation.y)

            yaw = quaternion_to_yaw(
                tf_msg.transform.rotation
            )

            return x, y, yaw

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
        # A visible tag should now be approached instead of only stopping.
        self.nav_planning(policy, GOAL_APPROACH_STANDOFF_M)

    def execute_goal(self, policy):
        self.nav_planning(policy, GOAL_APPROACH_STANDOFF_M)

    def execute_return_to_best_energy_bank(self, policy):
        self.nav_planning(policy, ENERGY_APPROACH_STANDOFF_M)

    def nav_planning(
        self,
        policy,
        approach_standoff_distance,
    ):
        """
        Build two standoff poses around the remembered tag, ask Nav2 to plan
        to both, and execute the shortest reachable path.

        Candidate A lies on the side from which the tag was successfully seen.
        Candidate B lies on the opposite side. Both poses face the tag.
        """
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

        if self.active_goal or self.planning_in_progress:
            self.get_logger().debug(
                "Navigation or candidate planning is already active. "
                "Ignoring repeated policy."
            )
            return

        if not self.goal_interval_ok():
            self.get_logger().debug(
                "Ignoring policy because minimum goal interval has not elapsed."
            )
            return

        target_x = float(policy.target_x_map)
        target_y = float(policy.target_y_map)
        observation_x = float(policy.last_seen_robot_x_map)
        observation_y = float(policy.last_seen_robot_y_map)

        robot_position = self.get_robot_position_from_tf()
        if robot_position is None:
            self.publish_simple_outcome(
                policy,
                False,
                False,
                "no_current_robot_tf_pose",
            )
            return

        robot_x, robot_y = robot_position

        values = (
            target_x,
            target_y,
            observation_x,
            observation_y,
            robot_x,
            robot_y,
            float(approach_standoff_distance),
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error(
                f"Approach geometry contains non-finite values: {values}"
            )
            self.publish_simple_outcome(
                policy,
                False,
                False,
                "invalid_approach_geometry",
            )
            return

        standoff = max(
            float(approach_standoff_distance),
            MIN_DISTANCE_EPSILON_M,
        )

        axis_x, axis_y, axis_source = self.get_approach_axis(
            policy=policy,
            target_x=target_x,
            target_y=target_y,
            observation_x=observation_x,
            observation_y=observation_y,
            robot_x=robot_x,
            robot_y=robot_y,
        )

        if axis_x is None or axis_y is None:
            self.get_logger().error(
                "Could not calculate a valid approach axis for the tag."
            )
            self.publish_simple_outcome(
                policy,
                False,
                False,
                "invalid_tag_approach_axis",
            )
            return

        candidate_specs = (
            ("observation_side", 1.0),
            ("opposite_side", -1.0),
        )

        candidates = []
        for name, sign in candidate_specs:
            goal_x = target_x + sign * standoff * axis_x
            goal_y = target_y + sign * standoff * axis_y
            goal_yaw = math.atan2(
                target_y - goal_y,
                target_x - goal_x,
            )

            candidates.append(
                {
                    "name": name,
                    "x": goal_x,
                    "y": goal_y,
                    "yaw": goal_yaw,
                    "pose": self.make_pose_stamped(
                        goal_x,
                        goal_y,
                        goal_yaw,
                    ),
                    "valid": False,
                    "path_length": math.inf,
                }
            )

        self.current_policy = policy
        self.energy_before_policy = self.get_current_energy()
        self.reward_before_policy = self.get_current_reward()

        self.candidate_plan_generation += 1
        generation = self.candidate_plan_generation
        self.planning_in_progress = True
        self.candidate_plan_candidates = candidates
        self.candidate_plan_index = 0
        self.selected_candidate_name = None

        self.get_logger().info(
            "Checking tag approach candidates with ComputePathToPose: "
            f"tag=({target_x:.3f}, {target_y:.3f}), "
            f"standoff={standoff:.3f} m, "
            f"axis=({axis_x:.3f}, {axis_y:.3f}) from {axis_source}; "
            f"A=({candidates[0]['x']:.3f}, {candidates[0]['y']:.3f}), "
            f"B=({candidates[1]['x']:.3f}, {candidates[1]['y']:.3f})"
        )

        if not self.compute_path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 ComputePathToPose action server not available"
            )
            self.finish_candidate_planning_failure(
                policy,
                "compute_path_server_unavailable",
            )
            return

        self.request_next_candidate_path(generation)

    def get_approach_axis(
        self,
        policy,
        target_x,
        target_y,
        observation_x,
        observation_y,
        robot_x,
        robot_y,
    ):
        """
        Return a unit vector pointing from the tag toward known free space.

        Prefer explicit map-frame normal fields if they are added to the policy
        later. With the current message, use the ray from the tag to the robot
        pose that successfully observed it. This is safer than treating the
        tag frame's ordinary Euler yaw as its wall-normal direction.
        """
        normal_field_pairs = (
            ("target_normal_x_map", "target_normal_y_map"),
            ("normal_x_map", "normal_y_map"),
        )

        for x_field, y_field in normal_field_pairs:
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

        dx = observation_x - target_x
        dy = observation_y - target_y
        magnitude = math.hypot(dx, dy)

        if magnitude > MIN_DISTANCE_EPSILON_M:
            return (
                dx / magnitude,
                dy / magnitude,
                "last_successful_observation_pose",
            )

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

    def make_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        qz, qw = yaw_to_quaternion_z_w(yaw)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def request_next_candidate_path(self, generation):
        if (
            generation != self.candidate_plan_generation
            or not self.planning_in_progress
        ):
            return

        if self.candidate_plan_index >= len(
            self.candidate_plan_candidates
        ):
            self.select_and_execute_candidate(generation)
            return

        index = self.candidate_plan_index
        candidate = self.candidate_plan_candidates[index]

        # Refresh the timestamp immediately before asking the planner.
        candidate["pose"].header.stamp = (
            self.get_clock().now().to_msg()
        )

        path_goal = ComputePathToPose.Goal()
        path_goal.goal = candidate["pose"]
        path_goal.planner_id = ""
        path_goal.use_start = False

        self.get_logger().info(
            f"Planning candidate {index + 1}/"
            f"{len(self.candidate_plan_candidates)} "
            f"{candidate['name']}: "
            f"({candidate['x']:.3f}, {candidate['y']:.3f}, "
            f"yaw={candidate['yaw']:.3f})"
        )

        send_future = self.compute_path_client.send_goal_async(
            path_goal
        )
        send_future.add_done_callback(
            lambda future, g=generation, i=index:
            self.candidate_path_goal_response_callback(
                future,
                g,
                i,
            )
        )

    def candidate_path_goal_response_callback(
        self,
        future,
        generation,
        index,
    ):
        if (
            generation != self.candidate_plan_generation
            or not self.planning_in_progress
        ):
            return

        try:
            goal_handle = future.result()
        except Exception as ex:
            self.get_logger().error(
                f"Candidate {index} path request failed: {ex}"
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="request_exception",
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(
                f"Candidate {index} ComputePathToPose goal was rejected"
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="goal_rejected",
            )
            return

        self.active_path_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, g=generation, i=index:
            self.candidate_path_result_callback(
                future,
                g,
                i,
            )
        )

    def candidate_path_result_callback(
        self,
        future,
        generation,
        index,
    ):
        if (
            generation != self.candidate_plan_generation
            or not self.planning_in_progress
        ):
            return

        self.active_path_goal_handle = None

        try:
            wrapped_result = future.result()
        except Exception as ex:
            self.get_logger().error(
                f"Candidate {index} path result failed: {ex}"
            )
            self.record_candidate_path_result(
                generation,
                index,
                valid=False,
                path_length=math.inf,
                reason="result_exception",
            )
            return

        succeeded = (
            wrapped_result.status
            == GoalStatus.STATUS_SUCCEEDED
        )

        path = wrapped_result.result.path
        path_has_poses = len(path.poses) > 0
        valid = bool(succeeded and path_has_poses)

        path_length = (
            self.compute_path_length(path)
            if valid
            else math.inf
        )

        reason = (
            "reachable"
            if valid
            else f"status_{wrapped_result.status}_empty_{not path_has_poses}"
        )

        self.record_candidate_path_result(
            generation,
            index,
            valid=valid,
            path_length=path_length,
            reason=reason,
        )

    def record_candidate_path_result(
        self,
        generation,
        index,
        valid,
        path_length,
        reason,
    ):
        if (
            generation != self.candidate_plan_generation
            or not self.planning_in_progress
        ):
            return

        candidate = self.candidate_plan_candidates[index]
        candidate["valid"] = bool(valid)
        candidate["path_length"] = float(path_length)

        if valid:
            self.get_logger().info(
                f"Candidate {candidate['name']} is reachable; "
                f"path_length={path_length:.3f} m"
            )
        else:
            self.get_logger().warn(
                f"Candidate {candidate['name']} is not usable: {reason}"
            )

        self.candidate_plan_index = index + 1
        self.request_next_candidate_path(generation)

    @staticmethod
    def compute_path_length(path):
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

    def select_and_execute_candidate(self, generation):
        if (
            generation != self.candidate_plan_generation
            or not self.planning_in_progress
        ):
            return

        valid_candidates = [
            candidate
            for candidate in self.candidate_plan_candidates
            if candidate["valid"]
        ]

        self.planning_in_progress = False
        self.active_path_goal_handle = None

        if not valid_candidates:
            policy = self.current_policy
            self.get_logger().error(
                "Neither tag approach candidate is reachable."
            )

            if policy is not None:
                self.publish_simple_outcome(
                    policy,
                    False,
                    False,
                    "no_reachable_tag_approach_candidate",
                )

            self.current_policy = None
            self.candidate_plan_candidates = []
            return

        selected = min(
            valid_candidates,
            key=lambda candidate: candidate["path_length"],
        )
        self.selected_candidate_name = selected["name"]

        self.get_logger().info(
            f"Selected {selected['name']} approach candidate: "
            f"goal=({selected['x']:.3f}, {selected['y']:.3f}, "
            f"yaw={selected['yaw']:.3f}), "
            f"path_length={selected['path_length']:.3f} m"
        )

        self.send_navigation_goal(selected)

    def send_navigation_goal(self, selected):
        policy = self.current_policy
        if policy is None:
            return

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
            self.current_policy = None
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = selected["pose"]
        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.active_goal = True
        self.last_goal_time = self.get_clock().now()

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            self.nav2_goal_response_callback
        )

    def finish_candidate_planning_failure(self, policy, status):
        self.candidate_plan_generation += 1
        self.planning_in_progress = False
        self.active_path_goal_handle = None
        self.candidate_plan_candidates = []

        self.publish_simple_outcome(
            policy,
            False,
            False,
            status,
        )
        self.current_policy = None

    def nav2_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as ex:
            self.get_logger().error(f"Failed to send Nav2 goal: {ex}")
            self.active_goal = False

            if self.current_policy is not None:
                self.publish_simple_outcome(
                    self.current_policy,
                    False,
                    False,
                    "nav2_goal_send_failed",
                )

            self.current_policy = None
            self.selected_candidate_name = None
            return

        if goal_handle is None or not goal_handle.accepted:
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
            self.selected_candidate_name = None
            return

        self.active_goal_handle = goal_handle

        self.get_logger().info(
            "Nav2 approach goal accepted"
        )

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
        except Exception as ex:
            self.get_logger().error(
                f"Failed to receive Nav2 result: {ex}"
            )
            status = -1

        success = status == GoalStatus.STATUS_SUCCEEDED

        if success:
            status_text = (
                "tag_approach_pose_reached_"
                f"{self.selected_candidate_name or 'candidate'}"
            )
            self.stop_robot()
        else:
            status_text = f"nav2_goal_finished_status_{status}"

        self.get_logger().info(status_text)

        self.active_goal = False
        self.active_goal_handle = None

        if self.current_policy is not None:
            self.publish_policy_outcome(
                self.current_policy,
                started=True,
                finished=True,
                success=success,
                status=status_text,
            )

        self.current_policy = None
        self.selected_candidate_name = None
        self.candidate_plan_candidates = []

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
        # Invalidate any ComputePathToPose callbacks still in flight.
        self.candidate_plan_generation += 1
        self.planning_in_progress = False
        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0

        if self.active_path_goal_handle is not None:
            self.active_path_goal_handle.cancel_goal_async()
            self.active_path_goal_handle = None

        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
            self.active_goal_handle = None

        self.active_goal = False
        self.current_policy = None
        self.selected_candidate_name = None

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