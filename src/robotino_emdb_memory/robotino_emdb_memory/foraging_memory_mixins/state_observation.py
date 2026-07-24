"""State publication, observations, and recharge authorization."""

import math
import statistics

from std_msgs.msg import Int32

from robotino_emdb_interfaces.msg import RobotinoForagingState, RobotinoTag

TAG_POSITION_WINDOW_SIZE = 15
TAG_POSITION_MIN_SAMPLES = 5
TAG_POSITION_OUTLIER_GATE_M = 0.45
TAG_POSITION_EMA_ALPHA = 0.25

class StateObservationMixin:
    """Mixin extracted from RobotinoForagingMemory."""

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

        # A drained renewable resource becomes eligible for a future visit only
        # after Robotino has physically departed from its interaction area.
        self.maybe_rearm_blocked_recharge(
            state.robot_x_map,
            state.robot_y_map,
        )

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
        self.observe_resource_bank(tag_id)

        # Merely seeing or passing an energy bank must never consume it.
        # Transfer starts only after the active return-to-energy policy has
        # reached its Nav2 goal and explicitly authorizes this exact tag ID.
        if (
            tag_id == self.active_recharge_target_id
            and tag_id != self.blocked_recharge_target_id
        ):
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
            # This is the last amount Robotino has observed, not a
            # time-predicted value derived from the regeneration rate.
            "resource_remaining": float(semantics["capacity"]),
            "collection_rate": float(semantics["collection_rate"]),
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
        supply = self.evaluate_bank_supply(remembered, resource_remaining)
        # Selection and an in-progress interaction use different gates.
        # An insufficient bank is not selectable again, but an already
        # authorized interaction may drain its remaining energy to the empty
        # epsilon so the executor can complete cleanly.
        authorized_partial_transfer = bool(
            tag_id == self.active_recharge_target_id
            and tag_id != self.blocked_recharge_target_id
            and supply["contains_energy"]
        )
        resource_available = bool(
            remembered["is_energy_bank"]
            and (supply["actionable"] or authorized_partial_transfer)
        )

        remembered["last_supply_reason"] = supply["reason"]
        remembered["last_supply_required"] = float(supply["required"])
        remembered["last_supply_deliverable"] = float(supply["deliverable"])

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

        # These fields are optional so the node remains compatible with the
        # current RobotinoForagingState message definition.
        self.set_if_available(
            state,
            "resource_required_for_recovery",
            float(supply["required"]),
        )
        self.set_if_available(
            state,
            "resource_deliverable_energy",
            float(supply["deliverable"]),
        )
        self.set_if_available(
            state,
            "resource_sufficient_for_recovery",
            bool(supply["actionable"]),
        )

        # Compatibility only. Official e-MDB Goal/Drive nodes own reward.
        state.novelty_reward = 0.0
        state.energy_reward = 0.0
        state.goal_reward = 0.0
        state.total_reward = 0.0

        state.goal_satisfied = bool(self.goal_satisfied)
        self.set_if_available(state, "goal_known", self.goal_known())
        self.fill_best_energy_bank(state)

    def maybe_rearm_blocked_recharge(self, robot_x, robot_y):
        """Allow a drained renewable bank to be used on a later visit.

        Clearing the executor authorization does not rearm the bank by itself;
        otherwise a stationary Robotino could immediately consume regenerated
        crumbs.  Rearming requires physical departure beyond the configured
        distance from the remembered tag position.
        """
        blocked_tag_id = int(self.blocked_recharge_target_id)
        if blocked_tag_id < 0:
            return False

        remembered = self.memory.get(blocked_tag_id)
        if remembered is None:
            self.blocked_recharge_target_id = -1
            return True

        try:
            tag_x = float(remembered["tag_x_map"])
            tag_y = float(remembered["tag_y_map"])
            distance = math.hypot(float(robot_x) - tag_x, float(robot_y) - tag_y)
        except (KeyError, TypeError, ValueError):
            return False

        if not math.isfinite(distance):
            return False
        if distance <= self.recharge_rearm_distance:
            return False

        self.blocked_recharge_target_id = -1
        self.get_logger().info(
            "Renewable bank rearmed for a future visit after Robotino left "
            "its vicinity: "
            f"tag_id={blocked_tag_id}, distance={distance:.3f}m, "
            f"threshold={self.recharge_rearm_distance:.3f}m."
        )
        return True

    def recharge_target_callback(self, msg: Int32):
        """Enable transfer for a nonempty bank selected by the executor."""
        requested_tag_id = int(msg.data)
        previous_tag_id = self.active_recharge_target_id

        if requested_tag_id < 0:
            self.active_recharge_target_id = -1
            if previous_tag_id >= 0:
                self.get_logger().info(
                    f"Recharge authorization cleared for tag {previous_tag_id}."
                )
            if self.blocked_recharge_target_id >= 0:
                self.get_logger().info(
                    "Drained bank remains unavailable for the current visit "
                    "until Robotino leaves its vicinity: "
                    f"tag_id={self.blocked_recharge_target_id}."
                )
            return

        if requested_tag_id == self.blocked_recharge_target_id:
            self.get_logger().debug(
                "Ignoring repeated recharge authorization for exhausted tag "
                f"{requested_tag_id} until a -1 clear command is received."
            )
            return

        resource = self.resource_truth.get(requested_tag_id)
        remembered = self.memory.get(requested_tag_id)
        if resource is None or remembered is None:
            self.get_logger().warn(
                "Ignoring recharge authorization for unknown or non-energy "
                f"tag {requested_tag_id}."
            )
            self.active_recharge_target_id = -1
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.update_resource_bank(requested_tag_id, now_sec)
        self.observe_resource_bank(requested_tag_id)
        supply = self.evaluate_bank_supply(
            remembered,
            resource.get("remaining", 0.0),
        )
        remembered["last_supply_reason"] = supply["reason"]
        remembered["last_supply_required"] = float(supply["required"])
        remembered["last_supply_deliverable"] = float(supply["deliverable"])

        if not supply["contains_energy"]:
            self.active_recharge_target_id = -1
            self.blocked_recharge_target_id = requested_tag_id
            self.get_logger().warn(
                "Rejected recharge authorization for an empty bank: "
                f"tag_id={requested_tag_id}, "
                f"remaining={supply['remaining']:.3f}."
            )
            self.fill_best_energy_bank(self.latest_state)
            self.publish_current_state()
            return

        if requested_tag_id == previous_tag_id:
            return

        self.active_recharge_target_id = requested_tag_id
        # Start timing from authorization, not from an earlier sighting.
        resource["last_collection_time"] = now_sec
        authorization_mode = (
            "full_recovery"
            if supply["actionable"]
            else "partial_then_leave"
        )
        self.get_logger().info(
            "Recharge authorized after navigation success: "
            f"tag_id={requested_tag_id}, mode={authorization_mode}, "
            f"remaining={float(resource['remaining']):.3f}, "
            f"deliverable={supply['deliverable']:.3f}, "
            f"required={supply['required']:.3f}, "
            f"robot_energy={self.robot_energy:.3f}."
        )

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