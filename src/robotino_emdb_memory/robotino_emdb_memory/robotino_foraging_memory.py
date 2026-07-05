import math

import rclpy
from rclpy.node import Node

from robotino_emdb_interfaces.msg import RobotinoTag, RobotinoForagingState


class RobotinoForagingMemory(Node):
    def __init__(self):
        super().__init__("robotino_foraging_memory")

        self.declare_parameter("input_topic", "/robotino/emdb/tag_observation")
        self.declare_parameter("output_topic", "/robotino/emdb/foraging_state")

        self.declare_parameter("initial_energy", 1.0)
        self.declare_parameter("energy_decay_per_second", 0.005)
        self.declare_parameter("low_energy_threshold", 0.35)

        self.declare_parameter("arrival_distance", 0.45)
        self.declare_parameter("same_tag_event_gap", 3.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.robot_energy = float(self.get_parameter("initial_energy").value)
        self.energy_decay_per_second = float(
            self.get_parameter("energy_decay_per_second").value
        )
        self.low_energy_threshold = float(
            self.get_parameter("low_energy_threshold").value
        )

        self.arrival_distance = float(self.get_parameter("arrival_distance").value)
        self.same_tag_event_gap = float(self.get_parameter("same_tag_event_gap").value)

        # Semantic meaning of each tag.
        # Later we can move this to a YAML file.
        self.tag_semantics = {
            0: {"type": "landmark", "energy": 0.0},
            1: {"type": "energy_bank", "energy": 0.30},
            2: {"type": "energy_bank", "energy": 0.50},
            3: {"type": "landmark", "energy": 0.0},
            4: {"type": "high_energy_bank", "energy": 0.80},
            5: {"type": "goal_marker", "energy": 0.0},
        }

        # Episodic/resource memory.
        # Key: tag_id
        # Value: remembered information about that tag.
        self.memory = {}

        self.last_update_time = self.get_clock().now()

        self.subscriber = self.create_subscription(
            RobotinoTag,
            self.input_topic,
            self.observation_callback,
            10,
        )

        self.publisher = self.create_publisher(
            RobotinoForagingState,
            self.output_topic,
            10,
        )

        self.energy_timer = self.create_timer(1.0, self.energy_decay_step)

        self.get_logger().info("Robotino foraging memory started")
        self.get_logger().info(f"Subscribing to: {self.input_topic}")
        self.get_logger().info(f"Publishing to: {self.output_topic}")

    def energy_decay_step(self):
        self.robot_energy -= self.energy_decay_per_second
        self.robot_energy = self.clamp(self.robot_energy, 0.0, 1.0)

    def observation_callback(self, msg: RobotinoTag):
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        state = RobotinoForagingState()
        state.header = msg.header

        state.valid = True
        state.visible = bool(msg.visible)

        state.robot_energy = float(self.robot_energy)
        state.energy_need = self.clamp(
            (self.low_energy_threshold - self.robot_energy) / self.low_energy_threshold,
            0.0,
            1.0,
        )

        if not msg.visible or msg.tag_id < 0:
            state.tag_id = -1
            state.tag_type = "none"
            state.first_time_seen = False
            state.known_tag = False
            state.resource_available = False
            state.best_energy_tag_id = self.get_best_energy_bank_id()
            self.fill_best_energy_bank(state)
            self.publisher.publish(state)
            return

        tag_id = int(msg.tag_id)
        semantics = self.tag_semantics.get(
            tag_id,
            {"type": "unknown", "energy": 0.0}
        )

        tag_type = semantics["type"]
        resource_capacity = float(semantics["energy"])
        is_energy_bank = resource_capacity > 0.0

        first_time_seen = tag_id not in self.memory

        if first_time_seen:
            self.memory[tag_id] = {
                "tag_type": tag_type,
                "first_seen_time": now_sec,
                "last_seen_time": now_sec,
                "times_seen": 1,

                "tag_x_map": float(msg.tag_x_map),
                "tag_y_map": float(msg.tag_y_map),
                "tag_yaw_map": float(msg.tag_yaw_map),

                "resource_capacity": resource_capacity,
                "resource_remaining": resource_capacity,
                "is_energy_bank": is_energy_bank,
            }

            novelty_reward = 1.0

        else:
            remembered = self.memory[tag_id]
            time_since_last_seen = now_sec - remembered["last_seen_time"]

            # Avoid counting every camera frame as a new episode.
            if time_since_last_seen >= self.same_tag_event_gap:
                remembered["times_seen"] += 1
                remembered["last_seen_time"] = now_sec

            # Smooth pose update. This lets repeated observations refine memory.
            alpha = 0.2
            remembered["tag_x_map"] = (
                (1.0 - alpha) * remembered["tag_x_map"] + alpha * float(msg.tag_x_map)
            )
            remembered["tag_y_map"] = (
                (1.0 - alpha) * remembered["tag_y_map"] + alpha * float(msg.tag_y_map)
            )
            remembered["tag_yaw_map"] = float(msg.tag_yaw_map)

            novelty_reward = 0.05 if time_since_last_seen >= self.same_tag_event_gap else 0.0

        remembered = self.memory[tag_id]

        energy_reward = 0.0

        resource_remaining = float(remembered["resource_remaining"])
        resource_available = is_energy_bank and resource_remaining > 0.0

        # Recharge if the robot is close enough to the energy bank.
        if (
            is_energy_bank
            and resource_available
            and msg.distance <= self.arrival_distance
        ):
            recharge = min(resource_remaining, 1.0 - self.robot_energy)

            if recharge > 0.0:
                self.robot_energy += recharge
                remembered["resource_remaining"] -= recharge
                energy_reward = recharge

        goal_reward = 1.0 if tag_type == "goal_marker" else 0.0
        goal_satisfied = tag_type == "goal_marker"

        # Simple reward model.
        total_reward = novelty_reward + energy_reward + goal_reward

        state.visible = True
        state.tag_id = tag_id
        state.tag_type = tag_type

        state.confidence = float(msg.confidence)
        state.distance = float(msg.distance)
        state.bearing = float(msg.bearing)

        state.tag_x_map = float(msg.tag_x_map)
        state.tag_y_map = float(msg.tag_y_map)
        state.tag_yaw_map = float(msg.tag_yaw_map)

        state.robot_x_map = float(msg.robot_x_map)
        state.robot_y_map = float(msg.robot_y_map)
        state.robot_yaw_map = float(msg.robot_yaw_map)

        state.first_time_seen = bool(first_time_seen)
        state.known_tag = not first_time_seen
        state.times_seen = int(remembered["times_seen"])
        state.time_since_last_seen = float(now_sec - remembered["last_seen_time"])

        state.is_energy_bank = bool(is_energy_bank)
        state.resource_capacity = float(resource_capacity)
        state.resource_remaining = float(remembered["resource_remaining"])
        state.resource_value = float(resource_capacity)
        state.resource_available = bool(
            is_energy_bank and remembered["resource_remaining"] > 0.0
        )

        state.robot_energy = float(self.robot_energy)
        state.energy_need = self.clamp(
            (self.low_energy_threshold - self.robot_energy) / self.low_energy_threshold,
            0.0,
            1.0,
        )

        state.novelty_reward = float(novelty_reward)
        state.energy_reward = float(energy_reward)
        state.goal_reward = float(goal_reward)
        state.total_reward = float(total_reward)

        self.fill_best_energy_bank(state)

        state.goal_satisfied = bool(goal_satisfied)

        self.publisher.publish(state)

    def get_best_energy_bank_id(self):
        best_id = -1
        best_score = 0.0

        for tag_id, data in self.memory.items():
            if not data["is_energy_bank"]:
                continue

            remaining = float(data["resource_remaining"])

            if remaining <= 0.0:
                continue

            score = remaining

            if score > best_score:
                best_score = score
                best_id = tag_id

        return best_id

    def fill_best_energy_bank(self, state: RobotinoForagingState):
        best_id = -1
        best_score = 0.0
        best_x = 0.0
        best_y = 0.0

        for tag_id, data in self.memory.items():
            if not data["is_energy_bank"]:
                continue

            remaining = float(data["resource_remaining"])

            if remaining <= 0.0:
                continue

            dx = float(data["tag_x_map"]) - float(state.robot_x_map)
            dy = float(data["tag_y_map"]) - float(state.robot_y_map)
            dist = math.sqrt(dx * dx + dy * dy)

            # Higher remaining resource and closer distance is better.
            score = remaining / (1.0 + dist)

            if score > best_score:
                best_score = score
                best_id = tag_id
                best_x = float(data["tag_x_map"])
                best_y = float(data["tag_y_map"])

        state.best_energy_tag_id = int(best_id)
        state.best_energy_x_map = float(best_x)
        state.best_energy_y_map = float(best_y)
        state.best_energy_score = float(best_score)

    def clamp(self, value, min_value, max_value):
        return float(max(min_value, min(max_value, value)))


def main(args=None):
    rclpy.init(args=args)

    node = RobotinoForagingMemory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()