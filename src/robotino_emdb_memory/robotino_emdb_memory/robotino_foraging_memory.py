#!/usr/bin/env python3
"""Robotino-specific semantic/resource memory for the GII e-MDB integration.

This node is NOT the e-MDB LTM. The official e-MDB LTM stores cognitive nodes,
contexts, P-Node points/anti-points, C-Nodes, and learned policy relations.

This node stores concrete Robotino facts that the generic architecture cannot
know by itself:
  * tag IDs and semantic types;
  * robust map positions and successful observation poses;
  * remaining resource amounts;
  * per-tag evidence for presence, reachability and recharge reliability.

It publishes RobotinoForagingState. A custom official e-MDB Perception node
normalizes that message and publishes /perception/foraging_state/value.

The official e-MDB MainLoop must create episodes and rewards. Therefore this
node no longer maintains its own episode list or calculates novelty/goal reward.
"""

import math
import statistics
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoTag,
)


# Robust tag-position estimation constants.
TAG_POSITION_WINDOW_SIZE = 15
TAG_POSITION_MIN_SAMPLES = 5
TAG_POSITION_OUTLIER_GATE_M = 0.45
TAG_POSITION_EMA_ALPHA = 0.25


class RobotinoForagingMemory(Node):
    """Maintain Robotino's factual resource registry and reliability evidence."""

    def __init__(self):
        super().__init__("robotino_foraging_memory")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter(
            "semantics_file",
            "/home/mike/eMDB_ws/src/robotino_emdb_memory/"
            "config/foraging_semantics.yaml",
        )
        self.declare_parameter(
            "memory_file",
            "~/.robotino_emdb/robotino_resource_memory.yaml",
        )
        self.declare_parameter("persist_memory", True)

        self.declare_parameter(
            "input_topic", "/robotino/emdb/tag_observation"
        )
        self.declare_parameter(
            "outcome_topic", "/robotino/emdb/policy_outcome"
        )
        self.declare_parameter(
            "output_topic", "/robotino/emdb/foraging_state"
        )

        self.declare_parameter("initial_energy", 1.0)
        self.declare_parameter("energy_decay_per_second", 0.01)
        self.declare_parameter("low_energy_threshold", 0.35)

        self.declare_parameter("arrival_distance", 1.2)
        self.declare_parameter("same_tag_event_gap", 3.0)
        self.declare_parameter("state_publish_rate_hz", 5.0)

        # Evidence weights for the three requested outcomes.
        self.declare_parameter("successful_recharge_weight", 2.0)
        self.declare_parameter("navigation_failure_weight", 1.0)
        self.declare_parameter("verified_absence_weight", 1.0)
        self.declare_parameter("missing_confidence_threshold", 0.35)
        self.declare_parameter("unreachable_confidence_threshold", 0.35)
        self.declare_parameter("missing_after_consecutive_failures", 2)
        self.declare_parameter("unreachable_after_consecutive_failures", 2)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.outcome_topic = str(self.get_parameter("outcome_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.robot_energy = float(self.get_parameter("initial_energy").value)
        self.energy_decay_per_second = float(
            self.get_parameter("energy_decay_per_second").value
        )
        self.low_energy_threshold = float(
            self.get_parameter("low_energy_threshold").value
        )

        self.arrival_distance = float(
            self.get_parameter("arrival_distance").value
        )
        self.same_tag_event_gap = float(
            self.get_parameter("same_tag_event_gap").value
        )
        self.state_publish_rate_hz = max(
            0.2, float(self.get_parameter("state_publish_rate_hz").value)
        )

        self.successful_recharge_weight = float(
            self.get_parameter("successful_recharge_weight").value
        )
        self.navigation_failure_weight = float(
            self.get_parameter("navigation_failure_weight").value
        )
        self.verified_absence_weight = float(
            self.get_parameter("verified_absence_weight").value
        )
        self.missing_confidence_threshold = float(
            self.get_parameter("missing_confidence_threshold").value
        )
        self.unreachable_confidence_threshold = float(
            self.get_parameter("unreachable_confidence_threshold").value
        )
        self.missing_after_consecutive_failures = int(
            self.get_parameter("missing_after_consecutive_failures").value
        )
        self.unreachable_after_consecutive_failures = int(
            self.get_parameter("unreachable_after_consecutive_failures").value
        )

        self.persist_memory = bool(
            self.get_parameter("persist_memory").value
        )
        self.memory_file = Path(
            str(self.get_parameter("memory_file").value)
        ).expanduser()

        # ------------------------------------------------------------------
        # Semantic and factual memory
        # ------------------------------------------------------------------
        self.semantics_file = str(
            self.get_parameter("semantics_file").value
        )
        self.tag_semantics = self.load_tag_semantics(self.semantics_file)
        self.warned_unknown_tag_ids = set()

        energy_tag_ids = sorted(
            tag_id
            for tag_id, semantics in self.tag_semantics.items()
            if bool(semantics.get("is_energy_bank", False))
        )
        self.get_logger().info(
            f"Configured energy-bank tag IDs: {energy_tag_ids}"
        )
        if not energy_tag_ids:
            self.get_logger().error(
                "No energy-bank tags are configured. "
                "Check foraging_semantics.yaml."
            )

        # Key: integer tag ID. Value: factual resource record.
        self.memory = {}
        self.load_resource_memory()

        self.goal_satisfied = False
        self.last_logged_best_energy_tag_id = None
        self.latest_state = self.create_empty_state()

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.observation_subscriber = self.create_subscription(
            RobotinoTag,
            self.input_topic,
            self.observation_callback,
            10,
        )
        self.outcome_subscriber = self.create_subscription(
            RobotinoPolicyOutcome,
            self.outcome_topic,
            self.outcome_callback,
            10,
        )
        self.publisher = self.create_publisher(
            RobotinoForagingState,
            self.output_topic,
            10,
        )

        self.energy_timer = self.create_timer(1.0, self.energy_decay_step)
        self.state_timer = self.create_timer(
            1.0 / self.state_publish_rate_hz,
            self.publish_current_state,
        )
        self.persistence_timer = self.create_timer(
            5.0,
            self.save_resource_memory,
        )

        self.get_logger().info("Robotino resource memory started")
        self.get_logger().info(f"Tag observations: {self.input_topic}")
        self.get_logger().info(f"Policy outcomes: {self.outcome_topic}")
        self.get_logger().info(f"Foraging state: {self.output_topic}")
        self.get_logger().info(f"Persistent memory: {self.memory_file}")

    # ==================================================================
    # Generic helpers
    # ==================================================================
    @staticmethod
    def clamp(value, min_value, max_value):
        return float(max(min_value, min(max_value, float(value))))

    @staticmethod
    def probability(positive, negative):
        total = float(positive) + float(negative)
        if total <= 0.0:
            return 0.5
        return float(positive) / total

    @staticmethod
    def set_if_available(message, field_name, value):
        """Set optional fields after RobotinoForagingState.msg is extended."""
        if hasattr(message, field_name):
            setattr(message, field_name, value)

    def energy_need(self):
        if self.low_energy_threshold <= 0.0:
            return 0.0
        return self.clamp(
            (self.low_energy_threshold - self.robot_energy)
            / self.low_energy_threshold,
            0.0,
            1.0,
        )

    # ==================================================================
    # Semantic configuration and persistence
    # ==================================================================
    def load_tag_semantics(self, semantics_file):
        path = Path(semantics_file)

        if not path.exists():
            self.get_logger().warn(
                f"Semantics file not found: {semantics_file}. Using defaults."
            )
            return {
                0: {
                    "type": "landmark",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
                1: {
                    "type": "low_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.30,
                    "collection_rate": 0.08,
                    "regen_rate": 0.0,
                },
                2: {
                    "type": "medium_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.50,
                    "collection_rate": 0.12,
                    "regen_rate": 0.0,
                },
                3: {
                    "type": "checkpoint",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
                4: {
                    "type": "high_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.80,
                    "collection_rate": 0.18,
                    "regen_rate": 0.0,
                },
                5: {
                    "type": "goal_marker",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
            }

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        semantics = {}
        for raw_id, tag_data in data.get("tags", {}).items():
            tag_id = int(raw_id)
            semantics[tag_id] = {
                "type": str(tag_data.get("type", "unknown")),
                "is_energy_bank": bool(
                    tag_data.get("is_energy_bank", False)
                ),
                "capacity": float(tag_data.get("capacity", 0.0)),
                "collection_rate": float(
                    tag_data.get("collection_rate", 0.0)
                ),
                "regen_rate": float(tag_data.get("regen_rate", 0.0)),
            }

        self.get_logger().info(
            f"Loaded tag semantics from: {semantics_file}"
        )
        return semantics

    def default_evidence(self):
        return {
            # A tag enters memory because it was visually detected.
            "presence_positive": 2.0,
            "presence_negative": 1.0,
            # Reachability/recharge begin uncertain.
            "reachability_positive": 1.0,
            "reachability_negative": 1.0,
            "recharge_positive": 1.0,
            "recharge_negative": 1.0,
            "navigation_attempts": 0,
            "navigation_successes": 0,
            "navigation_failures": 0,
            "verification_attempts": 0,
            "verified_absences": 0,
            "recharge_attempts": 0,
            "recharge_successes": 0,
            "consecutive_navigation_failures": 0,
            "consecutive_not_found": 0,
            "status": "UNVERIFIED",
            "last_outcome": "none",
            "last_failure_reason": "none",
        }

    def ensure_memory_schema(self, data):
        for key, value in self.default_evidence().items():
            data.setdefault(key, value)
        data.setdefault("position_samples", [])
        data["position_samples"] = [
            (float(sample[0]), float(sample[1]))
            for sample in data["position_samples"]
            if isinstance(sample, (list, tuple)) and len(sample) >= 2
        ]
        return data

    def apply_current_semantics(self, tag_id, data):
        """Migrate a remembered record to the currently loaded semantics.

        Earlier runs may have stored a tag as ``unknown`` or as a non-energy
        tag when the semantics path or tag IDs were wrong.  Without this
        migration, correcting the YAML would never make that persistent record
        eligible for best-bank selection.
        """
        semantics = self.tag_semantics.get(int(tag_id))
        if semantics is None:
            if int(tag_id) not in self.warned_unknown_tag_ids:
                self.warned_unknown_tag_ids.add(int(tag_id))
                self.get_logger().warn(
                    f"Tag {int(tag_id)} is not present in "
                    f"{self.semantics_file}; it cannot be an energy bank."
                )
            return self.ensure_memory_schema(data)

        previous_type = str(data.get("tag_type", "unknown"))
        previous_is_bank = bool(data.get("is_energy_bank", False))

        data["tag_type"] = str(semantics["type"])
        data["is_energy_bank"] = bool(semantics["is_energy_bank"])
        data["resource_capacity"] = float(semantics["capacity"])
        data["collection_rate"] = float(semantics["collection_rate"])
        data["regen_rate"] = float(semantics["regen_rate"])
        data.setdefault("resource_remaining", float(semantics["capacity"]))

        # Clamp old persisted values to the current configured capacity.
        capacity = max(0.0, float(data["resource_capacity"]))
        data["resource_remaining"] = self.clamp(
            data.get("resource_remaining", capacity),
            0.0,
            capacity,
        )

        if (
            previous_type != data["tag_type"]
            or previous_is_bank != data["is_energy_bank"]
        ):
            self.get_logger().warn(
                f"Migrated tag {int(tag_id)} semantics: "
                f"type {previous_type!r} -> {data['tag_type']!r}, "
                f"is_energy_bank {previous_is_bank} -> "
                f"{data['is_energy_bank']}"
            )

        return self.ensure_memory_schema(data)

    def load_resource_memory(self):
        if not self.persist_memory or not self.memory_file.exists():
            return

        try:
            with self.memory_file.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}

            raw_tags = payload.get("tags", {})
            for raw_id, data in raw_tags.items():
                tag_id = int(raw_id)
                record = self.apply_current_semantics(
                    tag_id, dict(data)
                )

                # ROS simulation time may restart between runs. Keep the facts
                # and evidence, but restart transient time references.
                record["last_seen_time"] = 0.0
                record["last_detection_time"] = 0.0
                record["last_resource_update_time"] = 0.0
                record.pop("last_collection_time", None)
                self.memory[tag_id] = record

            self.get_logger().info(
                f"Loaded {len(self.memory)} remembered tags"
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.get_logger().error(
                f"Could not load Robotino resource memory: {error}"
            )

    def save_resource_memory(self):
        if not self.persist_memory:
            return

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tags": {
                    int(tag_id): data
                    for tag_id, data in sorted(self.memory.items())
                }
            }
            temporary = self.memory_file.with_suffix(
                self.memory_file.suffix + ".tmp"
            )
            with temporary.open("w", encoding="utf-8") as file:
                yaml.safe_dump(payload, file, sort_keys=True)
            temporary.replace(self.memory_file)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.get_logger().error(
                f"Could not save Robotino resource memory: {error}"
            )

    # ==================================================================
    # State publication
    # ==================================================================
    def create_empty_state(self):
        state = RobotinoForagingState()
        state.valid = False
        state.visible = False
        state.tag_id = -1
        state.tag_type = "none"
        state.robot_energy = float(self.robot_energy)
        state.energy_need = self.energy_need()
        state.goal_satisfied = bool(self.goal_satisfied)
        self.set_if_available(state, "goal_known", self.goal_known())

        # Compatibility fields: rewards are now owned by e-MDB Goals/Drives.
        state.novelty_reward = 0.0
        state.energy_reward = 0.0
        state.goal_reward = 0.0
        state.total_reward = 0.0

        self.fill_best_energy_bank(state)
        return state

    def publish_current_state(self):
        state = self.latest_state
        state.header.stamp = self.get_clock().now().to_msg()
        state.robot_energy = float(self.robot_energy)
        state.energy_need = self.energy_need()
        state.goal_satisfied = bool(self.goal_satisfied)
        self.set_if_available(state, "goal_known", self.goal_known())
        self.fill_best_energy_bank(state)
        self.publisher.publish(state)

    def energy_decay_step(self):
        self.robot_energy = self.clamp(
            self.robot_energy - self.energy_decay_per_second,
            0.0,
            1.0,
        )

    def goal_known(self):
        return any(
            data.get("tag_type") == "goal_marker"
            for data in self.memory.values()
        )

    # ==================================================================
    # Tag observations
    # ==================================================================
    def observation_callback(self, msg: RobotinoTag):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        state = RobotinoForagingState()
        state.header = msg.header
        state.valid = True
        state.visible = bool(msg.visible)
        state.robot_energy = float(self.robot_energy)
        state.energy_need = self.energy_need()

        # Always carry the current robot pose, including empty detections.
        state.robot_x_map = float(msg.robot_x_map)
        state.robot_y_map = float(msg.robot_y_map)
        state.robot_yaw_map = float(msg.robot_yaw_map)

        if not msg.visible or msg.tag_id < 0:
            self.populate_no_visible_tag_state(state)
            self.latest_state = state
            self.publish_current_state()
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
            },
        )

        if tag_id not in self.tag_semantics:
            if tag_id not in self.warned_unknown_tag_ids:
                self.warned_unknown_tag_ids.add(tag_id)
                self.get_logger().warn(
                    f"Observed tag {tag_id}, but it has no entry in "
                    f"{self.semantics_file}. Current configured IDs: "
                    f"{sorted(self.tag_semantics)}"
                )

        first_time_seen = tag_id not in self.memory
        if first_time_seen:
            self.create_tag_memory(tag_id, semantics, msg, now_sec)
        else:
            # Always reapply current semantics so stale persistent records are
            # repaired after the YAML is corrected.
            self.apply_current_semantics(tag_id, self.memory[tag_id])
            self.update_tag_memory(tag_id, msg, now_sec)

        remembered = self.apply_current_semantics(
            tag_id, self.memory[tag_id]
        )

        # A new visual encounter is positive presence evidence, but camera
        # frames from the same continuous sighting are not counted repeatedly.
        time_since_last_seen = max(
            0.0, now_sec - float(remembered.get("last_seen_time", 0.0))
        )
        if first_time_seen or time_since_last_seen >= self.same_tag_event_gap:
            if not first_time_seen:
                remembered["times_seen"] = int(
                    remembered.get("times_seen", 0)
                ) + 1
                remembered["last_seen_time"] = now_sec
                remembered["presence_positive"] += 0.25
            remembered["consecutive_not_found"] = 0
            remembered["status"] = "ACTIVE"

        remembered["last_detection_time"] = now_sec
        self.update_resource_bank(tag_id, now_sec)
        self.collect_energy_from_bank(tag_id, msg, now_sec)

        self.populate_visible_tag_state(
            state,
            msg,
            tag_id,
            first_time_seen,
            remembered,
            now_sec,
        )

        self.latest_state = state
        self.publish_current_state()

    def create_tag_memory(self, tag_id, semantics, msg, now_sec):
        measurement_x = float(msg.tag_x_map)
        measurement_y = float(msg.tag_y_map)

        data = {
            "tag_type": str(semantics["type"]),
            "first_seen_time": now_sec,
            "last_seen_time": now_sec,
            "last_detection_time": now_sec,
            "last_resource_update_time": now_sec,
            "times_seen": 1,
            "position_samples": [(measurement_x, measurement_y)],
            "accepted_pose_samples": 1,
            "rejected_pose_samples": 0,
            "tag_x_map": measurement_x - 0.05,
            "tag_y_map": measurement_y,
            "tag_yaw_map": float(msg.tag_yaw_map),
            "last_seen_robot_x_map": float(msg.robot_x_map),
            "last_seen_robot_y_map": float(msg.robot_y_map),
            "last_seen_robot_yaw_map": float(msg.robot_yaw_map),
            "best_observation_confidence": float(msg.confidence),
            "best_observation_time": now_sec,
            "best_observation_distance": float(msg.distance),
            "best_observation_bearing": float(msg.bearing),
            "is_energy_bank": bool(semantics["is_energy_bank"]),
            "resource_capacity": float(semantics["capacity"]),
            "resource_remaining": float(semantics["capacity"]),
            "collection_rate": float(semantics["collection_rate"]),
            "regen_rate": float(semantics["regen_rate"]),
        }
        data.update(self.default_evidence())
        data["status"] = "ACTIVE"
        self.memory[tag_id] = data

        self.get_logger().info(
            f"Remembered new tag {tag_id} ({data['tag_type']}), "
            f"is_energy_bank={data['is_energy_bank']}, "
            f"resource_remaining={data['resource_remaining']:.3f}"
        )
        # Do not rely only on the five-second timer. A newly discovered tag is
        # important enough to persist immediately.
        self.save_resource_memory()

    def update_tag_memory(self, tag_id, msg, now_sec):
        remembered = self.ensure_memory_schema(self.memory[tag_id])
        measurement_x = float(msg.tag_x_map)
        measurement_y = float(msg.tag_y_map)
        samples = remembered.setdefault("position_samples", [])

        accept_measurement = True
        if len(samples) >= TAG_POSITION_MIN_SAMPLES:
            median_x = statistics.median(sample[0] for sample in samples)
            median_y = statistics.median(sample[1] for sample in samples)
            error_from_median = math.hypot(
                measurement_x - median_x,
                measurement_y - median_y,
            )
            accept_measurement = error_from_median <= TAG_POSITION_OUTLIER_GATE_M

            if not accept_measurement:
                remembered["rejected_pose_samples"] = int(
                    remembered.get("rejected_pose_samples", 0)
                ) + 1
                self.get_logger().warn(
                    f"Rejected tag {tag_id} position sample "
                    f"({measurement_x:.2f}, {measurement_y:.2f}); "
                    f"{error_from_median:.2f} m from rolling median"
                )

        if accept_measurement:
            samples.append((measurement_x, measurement_y))
            del samples[:-TAG_POSITION_WINDOW_SIZE]
            remembered["accepted_pose_samples"] = int(
                remembered.get("accepted_pose_samples", 0)
            ) + 1

            median_x = statistics.median(sample[0] for sample in samples)
            median_y = statistics.median(sample[1] for sample in samples)

            if len(samples) < TAG_POSITION_MIN_SAMPLES:
                remembered["tag_x_map"] = float(median_x)
                remembered["tag_y_map"] = float(median_y)
            else:
                alpha = TAG_POSITION_EMA_ALPHA
                remembered["tag_x_map"] = (
                    (1.0 - alpha) * float(remembered["tag_x_map"])
                    + alpha * median_x
                )
                remembered["tag_y_map"] = (
                    (1.0 - alpha) * float(remembered["tag_y_map"])
                    + alpha * median_y
                )

            remembered["tag_yaw_map"] = float(msg.tag_yaw_map)

        self.update_best_observation_pose(tag_id, msg, now_sec)

    def populate_no_visible_tag_state(self, state):
        state.visible = False
        state.tag_id = -1
        state.tag_type = "none"
        state.first_time_seen = False
        state.known_tag = False
        state.is_energy_bank = False
        state.resource_available = False
        state.resource_capacity = 0.0
        state.resource_remaining = 0.0
        state.resource_value = 0.0
        state.confidence = 0.0
        state.distance = 0.0
        state.bearing = 0.0

        state.novelty_reward = 0.0
        state.energy_reward = 0.0
        state.goal_reward = 0.0
        state.total_reward = 0.0
        state.goal_satisfied = bool(self.goal_satisfied)
        self.set_if_available(state, "goal_known", self.goal_known())
        self.fill_best_energy_bank(state)

    def populate_visible_tag_state(
        self,
        state,
        msg,
        tag_id,
        first_time_seen,
        remembered,
        now_sec,
    ):
        resource_remaining = float(remembered["resource_remaining"])
        resource_available = bool(
            remembered["is_energy_bank"] and resource_remaining > 0.0
        )

        state.visible = True
        state.tag_id = tag_id
        state.tag_type = str(remembered["tag_type"])
        state.confidence = float(msg.confidence)
        state.distance = float(msg.distance)
        state.bearing = float(msg.bearing)

        state.tag_x_map = float(remembered["tag_x_map"])
        state.tag_y_map = float(remembered["tag_y_map"])
        state.tag_yaw_map = float(remembered["tag_yaw_map"])

        state.last_seen_robot_x_map = float(
            remembered["last_seen_robot_x_map"]
        )
        state.last_seen_robot_y_map = float(
            remembered["last_seen_robot_y_map"]
        )
        state.last_seen_robot_yaw_map = float(
            remembered["last_seen_robot_yaw_map"]
        )

        state.first_time_seen = bool(first_time_seen)
        state.known_tag = not first_time_seen
        state.times_seen = int(remembered["times_seen"])
        state.time_since_last_seen = max(
            0.0, now_sec - float(remembered["last_seen_time"])
        )

        state.is_energy_bank = bool(remembered["is_energy_bank"])
        state.resource_capacity = float(remembered["resource_capacity"])
        state.resource_remaining = resource_remaining
        state.resource_value = float(remembered["resource_capacity"])
        state.resource_available = resource_available

        # Compatibility only. Official e-MDB Goal/Drive nodes own reward.
        state.novelty_reward = 0.0
        state.energy_reward = 0.0
        state.goal_reward = 0.0
        state.total_reward = 0.0

        state.goal_satisfied = bool(self.goal_satisfied)
        self.set_if_available(state, "goal_known", self.goal_known())
        self.fill_best_energy_bank(state)

    # ==================================================================
    # Policy outcomes -> factual per-tag evidence
    # ==================================================================
    def outcome_callback(self, msg: RobotinoPolicyOutcome):
        # Goal success is a mission fact, not "goal marker is visible".
        if (
            msg.policy_id == RobotinoPolicyOutcome.POLICY_GO_TO_GOAL
            and msg.policy_completed
            and msg.policy_success
        ):
            self.goal_satisfied = True

        if msg.target_type != RobotinoPolicyOutcome.TARGET_ENERGY_TAG:
            return
        if msg.target_id < 0:
            return

        tag_id = int(msg.target_id)
        if tag_id not in self.memory:
            self.get_logger().warn(
                f"Outcome received for unknown energy tag {tag_id}; ignoring"
            )
            return

        data = self.ensure_memory_schema(self.memory[tag_id])
        data["last_failure_reason"] = msg.failure_reason or "none"

        if self.is_complete_successful_recharge(msg):
            self.handle_successful_recharge(data)
        elif msg.navigation_result == RobotinoPolicyOutcome.NAV_FAILED:
            self.handle_navigation_failure(data)
        elif (
            msg.navigation_result == RobotinoPolicyOutcome.NAV_SUCCEEDED
            and msg.tag_result == RobotinoPolicyOutcome.TAG_NOT_FOUND
        ):
            self.handle_reached_but_tag_not_found(data)
        else:
            data["last_outcome"] = "UNSUPPORTED_OUTCOME"

        self.log_tag_reliability(tag_id, data)
        self.fill_best_energy_bank(self.latest_state)
        self.save_resource_memory()
        self.publish_current_state()

    @staticmethod
    def is_complete_successful_recharge(msg):
        return (
            msg.policy_id
            == RobotinoPolicyOutcome.POLICY_RETURN_TO_ENERGY
            and msg.policy_completed
            and msg.policy_success
            and msg.navigation_result
            == RobotinoPolicyOutcome.NAV_SUCCEEDED
            and msg.tag_result == RobotinoPolicyOutcome.TAG_FOUND
            and msg.recharge_attempted
            and msg.recharge_succeeded
            and msg.energy_after > msg.energy_before
        )

    def handle_successful_recharge(self, data):
        weight = self.successful_recharge_weight
        data["presence_positive"] += weight
        data["reachability_positive"] += weight
        data["recharge_positive"] += weight

        data["navigation_attempts"] += 1
        data["navigation_successes"] += 1
        data["verification_attempts"] += 1
        data["recharge_attempts"] += 1
        data["recharge_successes"] += 1
        data["consecutive_navigation_failures"] = 0
        data["consecutive_not_found"] = 0
        data["status"] = "ACTIVE"
        data["last_outcome"] = "SUCCESSFUL_RECHARGE"
        data["last_failure_reason"] = "none"

    def handle_navigation_failure(self, data):
        data["navigation_attempts"] += 1
        data["navigation_failures"] += 1
        data["consecutive_navigation_failures"] += 1
        data["reachability_negative"] += self.navigation_failure_weight
        data["last_outcome"] = "NAVIGATION_FAILED"

        # No presence or recharge update: Robotino did not inspect the tag.
        if (
            data["consecutive_navigation_failures"]
            >= self.unreachable_after_consecutive_failures
            or self.reachability_confidence(data)
            < self.unreachable_confidence_threshold
        ):
            data["status"] = "TEMPORARILY_UNREACHABLE"

    def handle_reached_but_tag_not_found(self, data):
        data["navigation_attempts"] += 1
        data["navigation_successes"] += 1
        data["verification_attempts"] += 1
        data["verified_absences"] += 1
        data["consecutive_not_found"] += 1
        data["consecutive_navigation_failures"] = 0

        # The observation pose was reachable, but the resource was absent.
        data["reachability_positive"] += self.verified_absence_weight
        data["presence_negative"] += self.verified_absence_weight
        data["last_outcome"] = "TAG_NOT_FOUND_AT_OBSERVATION_AREA"

        if (
            data["consecutive_not_found"]
            >= self.missing_after_consecutive_failures
            or self.presence_confidence(data)
            < self.missing_confidence_threshold
        ):
            data["status"] = "PROBABLY_MISSING"
        else:
            data["status"] = "UNVERIFIED"

    # ==================================================================
    # Reliability and target ranking
    # ==================================================================
    def presence_confidence(self, data):
        return self.probability(
            data["presence_positive"], data["presence_negative"]
        )

    def reachability_confidence(self, data):
        return self.probability(
            data["reachability_positive"],
            data["reachability_negative"],
        )

    def recharge_reliability(self, data):
        return self.probability(
            data["recharge_positive"], data["recharge_negative"]
        )

    def worthiness(self, data):
        return (
            self.presence_confidence(data)
            * self.reachability_confidence(data)
            * self.recharge_reliability(data)
        )

    def fill_best_energy_bank(self, state):
        """Rank banks by current foraging value multiplied by memory.

        foraging_score = remaining_resource / (1 + Euclidean distance)
        final_score = foraging_score * memory_worthiness

        Nav2 path length should replace Euclidean distance later when the target
        selector has a path-cost service available.
        """
        best_id = -1
        best_final_score = 0.0
        best_foraging_score = 0.0
        best_worthiness = 0.0
        best_presence = 0.0
        best_reachability = 0.0
        best_recharge = 0.0
        best_x = 0.0
        best_y = 0.0
        best_last_x = 0.0
        best_last_y = 0.0
        best_last_yaw = 0.0

        robot_x = float(getattr(state, "robot_x_map", 0.0))
        robot_y = float(getattr(state, "robot_y_map", 0.0))

        for tag_id, raw_data in self.memory.items():
            data = self.apply_current_semantics(tag_id, raw_data)
            if not data.get("is_energy_bank", False):
                continue

            remaining = float(data.get("resource_remaining", 0.0))
            if remaining <= 0.0:
                continue

            dx = float(data["tag_x_map"]) - robot_x
            dy = float(data["tag_y_map"]) - robot_y
            distance = math.hypot(dx, dy)

            foraging_score = remaining / (1.0 + distance)
            memory_worthiness = self.worthiness(data)
            final_score = foraging_score * memory_worthiness

            if final_score > best_final_score:
                best_final_score = final_score
                best_foraging_score = foraging_score
                best_worthiness = memory_worthiness
                best_presence = self.presence_confidence(data)
                best_reachability = self.reachability_confidence(data)
                best_recharge = self.recharge_reliability(data)
                best_id = int(tag_id)
                best_x = float(data["tag_x_map"])
                best_y = float(data["tag_y_map"])
                best_last_x = float(data["last_seen_robot_x_map"])
                best_last_y = float(data["last_seen_robot_y_map"])
                best_last_yaw = float(data["last_seen_robot_yaw_map"])

        state.best_energy_tag_id = int(best_id)
        state.best_energy_x_map = best_x
        state.best_energy_y_map = best_y
        state.best_energy_score = float(best_final_score)
        state.best_energy_last_seen_robot_x_map = best_last_x
        state.best_energy_last_seen_robot_y_map = best_last_y
        state.best_energy_last_seen_robot_yaw_map = best_last_yaw

        self.set_if_available(
            state, "best_energy_foraging_score", float(best_foraging_score)
        )
        self.set_if_available(
            state, "best_energy_presence_confidence", float(best_presence)
        )
        self.set_if_available(
            state,
            "best_energy_reachability_confidence",
            float(best_reachability),
        )
        self.set_if_available(
            state,
            "best_energy_recharge_reliability",
            float(best_recharge),
        )
        self.set_if_available(
            state, "best_energy_worthiness", float(best_worthiness)
        )

        if best_id != self.last_logged_best_energy_tag_id:
            self.last_logged_best_energy_tag_id = best_id
            if best_id >= 0:
                self.get_logger().info(
                    "Best energy bank -> tag %d | score=%.4f | "
                    "worthiness=%.3f | presence=%.3f | reachability=%.3f | "
                    "recharge=%.3f | observation_pose=(%.2f, %.2f, %.2f)"
                    % (
                        best_id,
                        best_final_score,
                        best_worthiness,
                        best_presence,
                        best_reachability,
                        best_recharge,
                        best_last_x,
                        best_last_y,
                        best_last_yaw,
                    )
                )
            else:
                self.get_logger().warn(
                    "No usable remembered energy bank. "
                    "Check observed tag IDs, semantics, and resource_remaining."
                )

    def log_tag_reliability(self, tag_id, data):
        self.get_logger().info(
            "Tag %d | outcome=%s | status=%s | presence=%.3f | "
            "reachability=%.3f | recharge=%.3f | worthiness=%.3f"
            % (
                tag_id,
                data["last_outcome"],
                data["status"],
                self.presence_confidence(data),
                self.reachability_confidence(data),
                self.recharge_reliability(data),
                self.worthiness(data),
            )
        )

    # ==================================================================
    # Observation-pose and resource simulation logic
    # ==================================================================
    def update_best_observation_pose(self, tag_id, msg, now_sec):
        if tag_id not in self.memory:
            return False

        current_confidence = float(msg.confidence)
        robot_x = float(msg.robot_x_map)
        robot_y = float(msg.robot_y_map)
        robot_yaw = float(msg.robot_yaw_map)

        if not math.isfinite(current_confidence):
            return False
        if not all(
            math.isfinite(value)
            for value in (robot_x, robot_y, robot_yaw)
        ):
            return False

        remembered = self.memory[tag_id]
        previous_best = float(
            remembered.get("best_observation_confidence", -math.inf)
        )
        if current_confidence <= previous_best:
            return False

        remembered["best_observation_confidence"] = current_confidence
        remembered["best_observation_time"] = float(now_sec)
        remembered["last_seen_robot_x_map"] = robot_x
        remembered["last_seen_robot_y_map"] = robot_y
        remembered["last_seen_robot_yaw_map"] = robot_yaw
        remembered["best_observation_distance"] = float(msg.distance)
        remembered["best_observation_bearing"] = float(msg.bearing)

        self.get_logger().info(
            f"Updated tag {tag_id} observation pose: "
            f"confidence {previous_best:.3f} -> {current_confidence:.3f}, "
            f"robot=({robot_x:.3f}, {robot_y:.3f}, yaw={robot_yaw:.3f})"
        )
        return True

    def update_resource_bank(self, tag_id, now_sec):
        if tag_id not in self.memory:
            return
        data = self.memory[tag_id]
        if not data.get("is_energy_bank", False):
            return

        last_time = float(data.get("last_resource_update_time", now_sec))
        if last_time <= 0.0:
            last_time = now_sec
        dt = max(0.0, now_sec - last_time)

        capacity = float(data["resource_capacity"])
        remaining = float(data["resource_remaining"])
        regen_rate = float(data["regen_rate"])

        if regen_rate > 0.0:
            remaining = min(capacity, remaining + regen_rate * dt)

        data["resource_remaining"] = remaining
        data["last_resource_update_time"] = now_sec

    def collect_energy_from_bank(self, tag_id, msg, now_sec):
        if tag_id not in self.memory:
            return 0.0
        data = self.memory[tag_id]
        if not data.get("is_energy_bank", False):
            return 0.0
        if msg.distance <= 0.05 or msg.distance > self.arrival_distance:
            return 0.0

        remaining = float(data["resource_remaining"])
        if remaining <= 0.0 or self.robot_energy >= 1.0:
            return 0.0

        last_time = float(data.get("last_collection_time", now_sec))
        dt = min(max(0.0, now_sec - last_time), 1.0)
        collection_rate = float(data["collection_rate"])
        amount_to_take = collection_rate * dt
        amount_taken = min(
            amount_to_take,
            remaining,
            1.0 - self.robot_energy,
        )

        data["last_collection_time"] = now_sec
        if amount_taken <= 0.0:
            return 0.0

        self.robot_energy = self.clamp(
            self.robot_energy + amount_taken,
            0.0,
            1.0,
        )
        data["resource_remaining"] = remaining - amount_taken
        return float(amount_taken)

    def destroy_node(self):
        self.save_resource_memory()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotinoForagingMemory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()