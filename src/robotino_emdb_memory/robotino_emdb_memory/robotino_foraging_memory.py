import math
import statistics
import yaml
from pathlib import Path

import rclpy
from rclpy.node import Node

from robotino_emdb_interfaces.msg import RobotinoTag, RobotinoForagingState

# Robust tag-position estimation constants.
TAG_POSITION_WINDOW_SIZE = 15
TAG_POSITION_MIN_SAMPLES = 5
TAG_POSITION_OUTLIER_GATE_M = 0.45
TAG_POSITION_EMA_ALPHA = 0.25

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
        
        measurement_x = float(msg.tag_x_map)
        measurement_y = float(msg.tag_y_map)

        # First Time Seen?
        if first_time_seen:
            self.memory[tag_id] = {
                "tag_type": tag_type,
                "first_seen_time": now_sec,
                "last_seen_time": now_sec,
                "last_detection_time": now_sec,
                "last_resource_update_time": now_sec,
                "times_seen": 1,

                # Keep a short sample window. The median prevents one bad first
                # frame from becoming the permanent remembered position.
                "position_samples": [(measurement_x, measurement_y)],
                "accepted_pose_samples": 1,
                "rejected_pose_samples": 0,

                "tag_x_map": measurement_x - 0.05,
                "tag_y_map": measurement_y,
                "tag_yaw_map": float(msg.tag_yaw_map),

                # These legacy "last_seen" fields now hold the robot pose
                # from the highest-confidence observation of this tag.
                "last_seen_robot_x_map": float(msg.robot_x_map),
                "last_seen_robot_y_map": float(msg.robot_y_map),
                "last_seen_robot_yaw_map": float(msg.robot_yaw_map),
                "best_observation_confidence": float(msg.confidence),
                "best_observation_time": now_sec,
                "best_observation_distance": float(msg.distance),
                "best_observation_bearing": float(msg.bearing),

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
            remembered["last_detection_time"] = now_sec

            # Avoid counting every camera frame as a new episode.
            new_episode = time_since_last_seen >= self.same_tag_event_gap
            if new_episode:
                remembered["times_seen"] += 1
                remembered["last_seen_time"] = now_sec

            samples = remembered.setdefault("position_samples", [])

            # Bootstrap freely for the first few frames so a bad first sample can
            # be outvoted by the following measurements. Once enough samples
            # exist, reject measurements that jump too far from the robust median.
            accept_measurement = True
            if len(samples) >= TAG_POSITION_MIN_SAMPLES:
                median_x = statistics.median(sample[0] for sample in samples)
                median_y = statistics.median(sample[1] for sample in samples)
                error_from_median = math.hypot(
                    measurement_x - median_x ,
                    measurement_y - median_y,
                )
                accept_measurement = (
                    error_from_median <= TAG_POSITION_OUTLIER_GATE_M
                )

                if not accept_measurement:
                    remembered["rejected_pose_samples"] = (
                        int(remembered.get("rejected_pose_samples", 0)) + 1
                    )
                    self.get_logger().warn(
                        f"Rejected tag {tag_id} position sample "
                        f"({measurement_x:.2f}, {measurement_y:.2f}); "
                        f"{error_from_median:.2f} m from rolling median"
                    )

            if accept_measurement:
                samples.append((measurement_x, measurement_y))
                del samples[:-TAG_POSITION_WINDOW_SIZE]

                remembered["accepted_pose_samples"] = (
                    int(remembered.get("accepted_pose_samples", 0)) + 1
                )

                median_x = statistics.median(sample[0] for sample in samples)
                median_y = statistics.median(sample[1] for sample in samples)

                # Before the sample set is stable, publish the median directly.
                # Afterwards, gently smooth movement of the robust estimate.
                if len(samples) < TAG_POSITION_MIN_SAMPLES:
                    remembered["tag_x_map"] = float(median_x)
                    remembered["tag_y_map"] = float(median_y)
                else:
                    alpha = TAG_POSITION_EMA_ALPHA
                    remembered["tag_x_map"] = (
                        (1.0 - alpha) * remembered["tag_x_map"]
                        + alpha * median_x
                    )
                    remembered["tag_y_map"] = (
                        (1.0 - alpha) * remembered["tag_y_map"]
                        + alpha * median_y
                    )

                remembered["tag_yaw_map"] = float(msg.tag_yaw_map)

                self.get_logger().info(
                    f"Tag {tag_id} robust map estimate: "
                    f"({remembered['tag_x_map']:.3f}, "
                    f"{remembered['tag_y_map']:.3f}) from "
                    f"{len(samples)} samples"
                )

            # The navigation observation pose is selected independently
            # from the rolling tag-position estimate. No confidence threshold
            # and no distance gate are used.
            self.update_best_observation_pose(tag_id, msg, now_sec)

            novelty_reward = 0.05 if new_episode else 0.0
        
        remembered = self.memory[tag_id]
        state.last_seen_robot_x_map = float(remembered["last_seen_robot_x_map"])
        state.last_seen_robot_y_map = float(remembered["last_seen_robot_y_map"])
        state.last_seen_robot_yaw_map = float(remembered["last_seen_robot_yaw_map"])
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
        """
        Publish the best remembered energy bank using only fields that exist
        in RobotinoForagingState.msg.

        best_energy_x_map and best_energy_y_map are the remembered map
        coordinates of the energy tag itself.
        """
        best_id = -1
        best_score = 0.0
        best_x = 0.0
        best_y = 0.0
        best_last_x = 0.0
        best_last_y = 0.0
        best_last_yaw = 0.0

        for tag_id, data in self.memory.items():
            if not data["is_energy_bank"]:
                continue

            remaining = float(data["resource_remaining"])

            if remaining <= 0.0:
                continue

            dx = float(data["tag_x_map"]) - float(state.robot_x_map)
            dy = float(data["tag_y_map"]) - float(state.robot_y_map)
            distance = math.hypot(dx, dy)

            # Prefer banks with more remaining resource and less travel.
            score = remaining / (1.0 + distance)

            if score > best_score:
                best_score = score
                best_id = int(tag_id)
                best_x = float(data["tag_x_map"])
                best_y = float(data["tag_y_map"])
                best_last_x = float(data["last_seen_robot_x_map"])
                best_last_y = float(data["last_seen_robot_y_map"])
                best_last_yaw = float(data["last_seen_robot_yaw_map"])

        state.best_energy_tag_id = int(best_id)
        state.best_energy_x_map = float(best_x)
        state.best_energy_y_map = float(best_y)
        state.best_energy_score = float(best_score)
        state.best_energy_last_seen_robot_x_map = float(best_last_x)
        state.best_energy_last_seen_robot_y_map = float(best_last_y)
        state.best_energy_last_seen_robot_yaw_map = float(best_last_yaw)

    def update_best_observation_pose(self, tag_id, msg, now_sec):
        """
        Save the robot pose associated with the highest-confidence detection.

        There is no confidence threshold and no distance condition. A new
        observation replaces the saved pose only when its confidence is
        strictly greater than the previous best for this tag.
        """
        if tag_id not in self.memory:
            return False

        current_confidence = float(msg.confidence)
        robot_x = float(msg.robot_x_map)
        robot_y = float(msg.robot_y_map)
        robot_yaw = float(msg.robot_yaw_map)

        if not math.isfinite(current_confidence):
            self.get_logger().warn(
                f"Ignoring tag {tag_id} observation with invalid confidence: "
                f"{current_confidence}"
            )
            return False

        if not all(
            math.isfinite(value)
            for value in (robot_x, robot_y, robot_yaw)
        ):
            self.get_logger().warn(
                f"Ignoring tag {tag_id} observation with invalid robot pose: "
                f"({robot_x}, {robot_y}, {robot_yaw})"
            )
            return False

        remembered = self.memory[tag_id]
        previous_best = float(
            remembered.get("best_observation_confidence", -math.inf)
        )

        if current_confidence <= previous_best:
            return False

        remembered["best_observation_confidence"] = current_confidence
        remembered["best_observation_time"] = float(now_sec)

        # Keep these existing names so the rest of the policy pipeline does
        # not need to change. Semantically they are now the best-confidence
        # observation pose, not simply the chronologically last pose.
        remembered["last_seen_robot_x_map"] = robot_x
        remembered["last_seen_robot_y_map"] = robot_y
        remembered["last_seen_robot_yaw_map"] = robot_yaw

        # Diagnostics only. Distance and bearing do not affect selection.
        remembered["best_observation_distance"] = float(msg.distance)
        remembered["best_observation_bearing"] = float(msg.bearing)

        self.get_logger().info(
            f"Updated tag {tag_id} navigation pose: "
            f"confidence {previous_best:.3f} -> {current_confidence:.3f}, "
            f"robot_pose=({robot_x:.3f}, {robot_y:.3f}, "
            f"yaw={robot_yaw:.3f})"
        )

        return True

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