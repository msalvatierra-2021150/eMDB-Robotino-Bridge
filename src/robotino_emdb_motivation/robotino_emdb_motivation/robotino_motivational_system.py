import rclpy
from rclpy.node import Node

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoMotivationState,
)


class RobotinoMotivationalSystem(Node):
    """
    Motivational layer for the Robotino foraging experiment.

    Input:
        /robotino/emdb/foraging_state

    Output:
        /robotino/emdb/motivation_state

    Purpose:
        Decide whether novelty, energy, or goal satisfaction is the
        dominant drive at the current moment.
    """

    DRIVE_NONE = 0
    DRIVE_NOVELTY = 1
    DRIVE_ENERGY = 2
    DRIVE_GOAL = 3

    POLICY_CONTINUE_EXPLORING = 0
    POLICY_INSPECT_VISIBLE_TAG = 1
    POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
    POLICY_SEARCH_FOR_ENERGY = 3
    POLICY_GOAL_REACHED = 4

    def __init__(self):
        super().__init__("robotino_motivational_system")

        self.declare_parameter("input_topic", "/robotino/emdb/foraging_state")
        self.declare_parameter("output_topic", "/robotino/emdb/motivation_state")

        self.declare_parameter("energy_critical_threshold", 0.75)
        self.declare_parameter("energy_high_threshold", 0.45)
        self.declare_parameter("minimum_bank_score", 0.01)

        self.declare_parameter("new_tag_novelty_boost", 1.0)
        self.declare_parameter("known_tag_novelty", 0.05)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.energy_critical_threshold = float(
            self.get_parameter("energy_critical_threshold").value
        )
        self.energy_high_threshold = float(
            self.get_parameter("energy_high_threshold").value
        )
        self.minimum_bank_score = float(
            self.get_parameter("minimum_bank_score").value
        )

        self.new_tag_novelty_boost = float(
            self.get_parameter("new_tag_novelty_boost").value
        )
        self.known_tag_novelty = float(
            self.get_parameter("known_tag_novelty").value
        )

        self.subscriber = self.create_subscription(
            RobotinoForagingState,
            self.input_topic,
            self.foraging_callback,
            10,
        )

        self.publisher = self.create_publisher(
            RobotinoMotivationState,
            self.output_topic,
            10,
        )

        self.get_logger().info("Robotino motivational system started")
        self.get_logger().info(f"Subscribing to: {self.input_topic}")
        self.get_logger().info(f"Publishing to: {self.output_topic}")

    def foraging_callback(self, msg: RobotinoForagingState):
        motivation = RobotinoMotivationState()
        motivation.header = msg.header
        motivation.valid = bool(msg.valid)

        if not msg.valid:
            motivation.dominant_drive_id = self.DRIVE_NONE
            motivation.dominant_drive = "none"
            motivation.suggested_policy_id = self.POLICY_CONTINUE_EXPLORING
            motivation.suggested_policy = "continue_exploring"
            motivation.suggested_goal = "keep_system_alive"
            self.publisher.publish(motivation)
            return

        novelty_drive = self.compute_novelty_drive(msg)
        energy_drive = self.compute_energy_drive(msg)
        goal_drive = self.compute_goal_drive(msg)

        motivation.novelty_drive = float(novelty_drive)
        motivation.energy_drive = float(energy_drive)
        motivation.goal_drive = float(goal_drive)

        exploration_priority = self.compute_exploration_priority(msg, novelty_drive, energy_drive)
        inspect_priority = self.compute_inspect_priority(msg, novelty_drive, energy_drive)
        return_priority = self.compute_return_priority(msg, energy_drive)
        search_energy_priority = self.compute_search_energy_priority(msg, energy_drive)

        motivation.exploration_priority = float(exploration_priority)
        motivation.inspect_priority = float(inspect_priority)
        motivation.return_priority = float(return_priority)
        motivation.search_energy_priority = float(search_energy_priority)

        policy_id, policy_name, drive_id, drive_name, suggested_goal = self.select_policy(
            msg,
            novelty_drive,
            energy_drive,
            goal_drive,
            exploration_priority,
            inspect_priority,
            return_priority,
            search_energy_priority,
        )

        motivation.suggested_policy_id = int(policy_id)
        motivation.suggested_policy = policy_name

        motivation.dominant_drive_id = int(drive_id)

        motivation.target_tag_id = int(msg.best_energy_tag_id)
        motivation.target_x_map = float(msg.best_energy_x_map)
        motivation.target_y_map = float(msg.best_energy_y_map)

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            motivation.last_seen_robot_x_map = float(msg.best_energy_last_seen_robot_x_map)
            motivation.last_seen_robot_y_map = float(msg.best_energy_last_seen_robot_y_map)
            motivation.last_seen_robot_yaw_map = float(msg.best_energy_last_seen_robot_yaw_map)
        else:
            motivation.last_seen_robot_x_map = float(msg.last_seen_robot_x_map)
            motivation.last_seen_robot_y_map = float(msg.last_seen_robot_y_map)
            motivation.last_seen_robot_yaw_map = float(msg.last_seen_robot_yaw_map)

        motivation.dominant_drive = drive_name

        motivation.suggested_goal = suggested_goal

        motivation.priority_confidence = self.compute_priority_confidence(
            exploration_priority,
            inspect_priority,
            return_priority,
            search_energy_priority,
        )

        motivation.should_interrupt_exploration = bool(
            policy_id in [
                self.POLICY_RETURN_TO_BEST_ENERGY_BANK,
                self.POLICY_GOAL_REACHED,
            ]
        )

        motivation.should_send_nav_goal = bool(
            policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK
        )

        self.publisher.publish(motivation)

    def compute_novelty_drive(self, msg: RobotinoForagingState):
        """
        Novelty is high when a tag is seen for the first time.
        Known repeated tags have low novelty.
        """
        if msg.first_time_seen:
            return self.clamp(max(msg.novelty_reward, self.new_tag_novelty_boost))

        if msg.visible and not msg.known_tag:
            return self.clamp(0.5)

        if msg.known_tag:
            return self.clamp(max(msg.novelty_reward, self.known_tag_novelty))

        return 0.0

    def compute_energy_drive(self, msg: RobotinoForagingState):
        """
        Energy drive is based on energy_need.

        If a known energy bank exists, energy need becomes more actionable.
        If no bank is known, the robot still has energy need, but the policy
        should become search_for_energy instead of return_to_bank.
        """
        energy_need = self.clamp(msg.energy_need)

        if msg.best_energy_tag_id >= 0:
            return energy_need

        return self.clamp(0.7 * energy_need)

    def compute_goal_drive(self, msg: RobotinoForagingState):
        """
        Goal drive is high if the goal marker or task objective is satisfied.
        Later this can be replaced by a richer goal manager.
        """
        if msg.goal_satisfied:
            return 1.0

        return self.clamp(msg.goal_reward)

    def compute_exploration_priority(self, msg, novelty_drive, energy_drive):
        """
        Exploration is useful when energy need is low or no better target exists.
        """
        if msg.energy_need >= self.energy_critical_threshold and msg.best_energy_tag_id >= 0:
            return 0.0

        return self.clamp((1.0 - msg.energy_need) * 0.6 + novelty_drive * 0.4)

    def compute_inspect_priority(self, msg, novelty_drive, energy_drive):
        """
        Inspecting is useful when a new tag is visible and energy is not critical.
        """
        if not msg.visible:
            return 0.0

        if msg.energy_need >= self.energy_critical_threshold and msg.best_energy_tag_id >= 0:
            return 0.0

        if msg.first_time_seen:
            return self.clamp(0.8 + 0.2 * novelty_drive)

        return self.clamp(0.2 * novelty_drive)

    def compute_return_priority(self, msg, energy_drive):
        """
        Returning is useful when energy is needed and a remembered bank exists.
        """
        if msg.best_energy_tag_id < 0:
            return 0.0

        if msg.best_energy_score < self.minimum_bank_score:
            return 0.0

        return self.clamp(energy_drive * (0.5 + 0.5 * msg.best_energy_score))

    def compute_search_energy_priority(self, msg, energy_drive):
        """
        Searching for energy is useful when energy is needed but no bank is known.
        """
        if msg.best_energy_tag_id >= 0:
            return 0.0

        return self.clamp(energy_drive)

    def select_policy(
        self,
        msg,
        novelty_drive,
        energy_drive,
        goal_drive,
        exploration_priority,
        inspect_priority,
        return_priority,
        search_energy_priority,
    ):
        """
        Select the suggested behavior.

        This is still rule-based. Later, the learning system can replace or tune
        these priorities using learned utility values.
        """

        if goal_drive >= 0.9 or msg.goal_satisfied:
            return (
                self.POLICY_GOAL_REACHED,
                "goal_reached",
                self.DRIVE_GOAL,
                "goal",
                "complete_task",
            )

        if msg.energy_need >= self.energy_critical_threshold and return_priority > 0.0:
            return (
                self.POLICY_RETURN_TO_BEST_ENERGY_BANK,
                "return_to_best_energy_bank",
                self.DRIVE_ENERGY,
                "energy",
                "recover_energy",
            )

        if msg.energy_need >= self.energy_high_threshold and search_energy_priority > 0.0:
            return (
                self.POLICY_SEARCH_FOR_ENERGY,
                "search_for_energy",
                self.DRIVE_ENERGY,
                "energy",
                "find_energy_bank",
            )

        if inspect_priority >= exploration_priority and inspect_priority > 0.0:
            return (
                self.POLICY_INSPECT_VISIBLE_TAG,
                "inspect_visible_tag",
                self.DRIVE_NOVELTY,
                "novelty",
                "inspect_new_semantic_event",
            )

        return (
            self.POLICY_CONTINUE_EXPLORING,
            "continue_exploring",
            self.DRIVE_NOVELTY,
            "novelty",
            "discover_new_information",
        )

    def compute_priority_confidence(self, *priorities):
        sorted_priorities = sorted([float(p) for p in priorities], reverse=True)

        if len(sorted_priorities) < 2:
            return self.clamp(sorted_priorities[0]) if sorted_priorities else 0.0

        best = sorted_priorities[0]
        second = sorted_priorities[1]

        return self.clamp(best - second)

    def clamp(self, value, min_value=0.0, max_value=1.0):
        return float(max(min_value, min(max_value, float(value))))


def main(args=None):
    rclpy.init(args=args)

    node = RobotinoMotivationalSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
