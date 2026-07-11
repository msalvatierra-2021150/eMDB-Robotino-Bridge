import math
import yaml
from pathlib import Path

import rclpy
from rclpy.node import Node

from robotino_emdb_interfaces.msg import RobotinoTag, RobotinoForagingState

class RobotinoForagingMemory(Node):
    def __init__(self):
        super().__init__("robotino_foraging_memory")

        self.declare_parameter(
            "semantics_file",
            "/home/mike/eMDB_ws/src/robotino_emdb_memory/config/foraging_semantics.yaml"
        )        

        self.declare_parameter("input_topic", "/robotino/emdb/tag_observation")
        self.declare_parameter("output_topic", "/robotino/emdb/foraging_state")

        self.declare_parameter("initial_energy", 1.0)
        self.declare_parameter("energy_decay_per_second", 0.01)
        self.declare_parameter("low_energy_threshold", 0.35)

        self.declare_parameter("arrival_distance", 1.2)
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
        self.semantics_file = self.get_parameter("semantics_file").value
        self.tag_semantics = self.load_tag_semantics(self.semantics_file)
        
        # Episodic/resource memory.
        # Key: tag_id
        # Value: remembered information about that tag.
        self.memory = {}
        self.episodes = []
        self.episode_count = 0

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

    def load_tag_semantics(self, semantics_file):
        path = Path(semantics_file)

        if not path.exists():
            self.get_logger().warn(
                f"Semantics file not found: {semantics_file}. Using defaults."
            )
            return {
                0: {"type": "landmark", "is_energy_bank": False, "capacity": 0.0, "collection_rate": 0.0, "regen_rate": 0.0},
                1: {"type": "low_energy_bank", "is_energy_bank": True, "capacity": 0.30, "collection_rate": 0.08, "regen_rate": 0.0},
                2: {"type": "medium_energy_bank", "is_energy_bank": True, "capacity": 0.50, "collection_rate": 0.12, "regen_rate": 0.0},
                3: {"type": "checkpoint", "is_energy_bank": False, "capacity": 0.0, "collection_rate": 0.0, "regen_rate": 0.0},
                4: {"type": "high_energy_bank", "is_energy_bank": True, "capacity": 0.80, "collection_rate": 0.18, "regen_rate": 0.0},
                5: {"type": "goal_marker", "is_energy_bank": False, "capacity": 0.0, "collection_rate": 0.0, "regen_rate": 0.0},
            }

        with open(path, "r") as file:
            data = yaml.safe_load(file) or {}

        semantics = {}

        for raw_id, tag_data in data.get("tags", {}).items():
            tag_id = int(raw_id)

            semantics[tag_id] = {
                "type": str(tag_data.get("type", "unknown")),
                "is_energy_bank": bool(tag_data.get("is_energy_bank", False)),
                "capacity": float(tag_data.get("capacity", 0.0)),
                "collection_rate": float(tag_data.get("collection_rate", 0.0)),
                "regen_rate": float(tag_data.get("regen_rate", 0.0)),
            }

        self.get_logger().info(f"Loaded tag semantics from: {semantics_file}")
        return semantics

    def energy_decay_step(self):
        self.robot_energy -= self.energy_decay_per_second
        self.robot_energy = self.clamp(self.robot_energy, 0.0, 1.0)
    
    #Observations
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

            state.robot_x_map = float(msg.robot_x_map)
            state.robot_y_map = float(msg.robot_y_map)
            state.robot_yaw_map = float(msg.robot_yaw_map)

            state.first_time_seen = False
            state.known_tag = False
            state.resource_available = False

            self.fill_best_energy_bank(state)

            self.publisher.publish(state)
            return

        tag_id = int(msg.tag_id)
        semantics = self.tag_semantics.get(
            tag_id,
            {
                "type": "unknown",
                "is_energy_bank": False,
                "capacity": 0.0,
                "collection_rate": 0.0,
                "regen_rate": 0.0,
            }
        )

        tag_type = semantics["type"]
        is_energy_bank = bool(semantics["is_energy_bank"])
        resource_capacity = float(semantics["capacity"])
        collection_rate = float(semantics["collection_rate"])
        regen_rate = float(semantics["regen_rate"])

        first_time_seen = tag_id not in self.memory
        
        #First Time Seen?
        if first_time_seen:
            self.memory[tag_id] = {
                "tag_type": tag_type,
                "first_seen_time": now_sec,
                "last_seen_time": now_sec,
                "last_resource_update_time": now_sec,
                "times_seen": 1,

                "tag_x_map": float(msg.tag_x_map),
                "tag_y_map": float(msg.tag_y_map),
                "tag_yaw_map": float(msg.tag_yaw_map),

                "last_seen_robot_x_map": float(msg.robot_x_map),
                "last_seen_robot_y_map": float(msg.robot_y_map),
                "last_seen_robot_yaw_map": float(msg.robot_yaw_map),

                "is_energy_bank": is_energy_bank,
                "resource_capacity": resource_capacity,
                "resource_remaining": resource_capacity,
                "collection_rate": collection_rate,
                "regen_rate": regen_rate,
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
        self.update_resource_bank(tag_id, now_sec)

        energy_reward = self.collect_energy_from_bank(tag_id, msg, now_sec)

        if first_time_seen:
            self.add_episode("NEW_TAG_DISCOVERED", tag_id, msg, novelty_reward)

        if first_time_seen and is_energy_bank:
            self.add_episode("ENERGY_BANK_DISCOVERED", tag_id, msg, novelty_reward)

        if energy_reward > 0.0:
            self.add_episode("ENERGY_COLLECTED", tag_id, msg, energy_reward)

        resource_remaining = float(remembered["resource_remaining"])
        resource_available = bool(
            remembered["is_energy_bank"] and resource_remaining > 0.0
        )

        goal_reward = 1.0 if tag_type == "goal_marker" else 0.0
        goal_satisfied = tag_type == "goal_marker"

        total_reward = novelty_reward + energy_reward + goal_reward

        state.visible = True
        state.tag_id = tag_id
        state.tag_type = tag_type

        state.confidence = float(msg.confidence)
        state.distance = float(msg.distance)
        state.bearing = float(msg.bearing)

        state.tag_x_map = float(remembered["tag_x_map"])
        state.tag_y_map = float(remembered["tag_y_map"])
        state.tag_yaw_map = float(remembered["tag_yaw_map"])

        state.robot_x_map = float(msg.robot_x_map)
        state.robot_y_map = float(msg.robot_y_map)
        state.robot_yaw_map = float(msg.robot_yaw_map)

        state.first_time_seen = bool(first_time_seen)
        state.known_tag = not first_time_seen
        state.times_seen = int(remembered["times_seen"])
        state.time_since_last_seen = float(now_sec - remembered["last_seen_time"])

        state.is_energy_bank = bool(remembered["is_energy_bank"])
        state.resource_capacity = float(remembered["resource_capacity"])
        state.resource_remaining = float(remembered["resource_remaining"])
        state.resource_value = float(remembered["resource_capacity"])
        state.resource_available = bool(resource_available)

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

    # Resource Bank logic
    def update_resource_bank(self, tag_id, now_sec):
        if tag_id not in self.memory:
            return

        data = self.memory[tag_id]

        if not data["is_energy_bank"]:
            return

        last_time = float(data.get("last_resource_update_time", now_sec))
        dt = max(0.0, now_sec - last_time)

        capacity = float(data["resource_capacity"])
        remaining = float(data["resource_remaining"])
        regen_rate = float(data["regen_rate"])

        if regen_rate > 0.0:
            remaining = min(capacity, remaining + regen_rate * dt)

        data["resource_remaining"] = remaining
        data["last_resource_update_time"] = now_sec

    # Energy from bank
    def collect_energy_from_bank(self, tag_id, msg, now_sec):
        if tag_id not in self.memory:
            return 0.0

        data = self.memory[tag_id]

        if not data["is_energy_bank"]:
            return 0.0

        if msg.distance <= 0.05:
            return 0.0

        if msg.distance > self.arrival_distance:
            return 0.0

        remaining = float(data["resource_remaining"])

        if remaining <= 0.0:
            return 0.0

        if self.robot_energy >= 1.0:
            return 0.0

        last_time = float(data.get("last_collection_time", now_sec))
        dt = max(0.0, now_sec - last_time)

        # Avoid huge energy jumps if the node was paused or started late.
        dt = min(dt, 1.0)

        collection_rate = float(data["collection_rate"])

        amount_to_take = collection_rate * dt

        amount_taken = min(
            amount_to_take,
            remaining,
            1.0 - self.robot_energy
        )

        if amount_taken <= 0.0:
            data["last_collection_time"] = now_sec
            return 0.0

        self.robot_energy = self.clamp(
            self.robot_energy + amount_taken,
            0.0,
            1.0
        )

        data["resource_remaining"] = remaining - amount_taken
        data["last_collection_time"] = now_sec

        return float(amount_taken)
    
    #Episodes
    def add_episode(self, event_type, tag_id, msg, reward):
        self.episode_count += 1

        episode = {
            "episode_id": self.episode_count,
            "time": self.get_clock().now().nanoseconds / 1e9,
            "event_type": event_type,
            "tag_id": int(tag_id),
            "robot_x_map": float(msg.robot_x_map),
            "robot_y_map": float(msg.robot_y_map),
            "robot_yaw_map": float(msg.robot_yaw_map),
            "tag_x_map": float(msg.tag_x_map),
            "tag_y_map": float(msg.tag_y_map),
            "distance": float(msg.distance),
            "bearing": float(msg.bearing),
            "confidence": float(msg.confidence),
            "robot_energy": float(self.robot_energy),
            "reward": float(reward),
        }

        self.episodes.append(episode)

        self.get_logger().info(
            f"Episode {self.episode_count}: {event_type}, "
            f"tag={tag_id}, reward={reward:.3f}, "
            f"energy={self.robot_energy:.3f}"
        )


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