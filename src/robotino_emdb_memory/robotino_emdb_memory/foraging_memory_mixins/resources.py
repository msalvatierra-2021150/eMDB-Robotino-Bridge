"""Private resource-world simulation and recovery-aware supply evaluation."""

import math


class ResourceSimulationMixin:
    """Manage hidden resource truth and determine whether a bank is usable."""

    def energy_required_for_recovery(self):
        """Return the net energy still needed to release recovery mode."""
        return max(
            0.0,
            float(self.resume_energy_threshold)
            - float(self.robot_energy)
            + float(self.recovery_supply_margin),
        )

    def estimated_deliverable_energy(self, data, remaining):
        """Estimate net robot-energy gain before the observed amount is drained.

        The bank transfers at ``collection_rate`` while the robot continuously
        loses ``energy_decay_per_second``.  Therefore an observed amount is not
        fully deliverable to the battery when charging takes time.
        """
        remaining = max(0.0, float(remaining))
        if remaining <= self.resource_empty_epsilon:
            return 0.0

        collection_rate = max(
            0.0,
            float(data.get("collection_rate", 0.0)),
        )
        if collection_rate <= 0.0:
            return 0.0

        decay_rate = max(0.0, float(self.energy_decay_per_second))
        net_rate = collection_rate - decay_rate
        if net_rate <= 0.0:
            return 0.0

        drain_time = remaining / collection_rate
        return self.clamp(net_rate * drain_time, 0.0, remaining)

    def evaluate_bank_supply(self, data, remaining=None):
        """Classify a bank against the Robotino's current recovery gap."""
        if remaining is None:
            remaining = data.get("resource_remaining", 0.0)

        try:
            remaining = float(remaining)
        except (TypeError, ValueError):
            remaining = 0.0
        if not math.isfinite(remaining):
            remaining = 0.0
        remaining = max(0.0, remaining)

        required = self.energy_required_for_recovery()
        deliverable = self.estimated_deliverable_energy(data, remaining)
        contains_energy = remaining > self.resource_empty_epsilon

        if not contains_energy:
            reason = "empty"
            actionable = False
        elif not self.require_full_recovery_supply:
            reason = "partial_supply_allowed"
            actionable = True
        elif deliverable + self.actionable_supply_epsilon >= required:
            reason = "sufficient_for_recovery"
            actionable = True
        else:
            reason = "insufficient_for_recovery"
            actionable = False

        return {
            "remaining": remaining,
            "required": required,
            "deliverable": deliverable,
            "contains_energy": contains_energy,
            "actionable": actionable,
            "reason": reason,
        }

    def update_resource_bank(self, tag_id, now_sec):
        """Advance hidden physical resource state using the YAML rate."""
        resource = self.resource_truth.get(int(tag_id))
        if resource is None:
            return

        last_time = float(resource.get("last_update_time", now_sec))
        if last_time <= 0.0:
            last_time = now_sec
        dt = max(0.0, now_sec - last_time)

        remaining = float(resource["remaining"])
        regen_rate = float(resource["regen_rate"])
        if regen_rate > 0.0:
            remaining = min(
                float(resource["capacity"]),
                remaining + regen_rate * dt,
            )

        resource["remaining"] = remaining
        resource["last_update_time"] = now_sec

    def observe_resource_bank(self, tag_id):
        """Copy the observed amount into memory unless this visit is drained.

        Hidden resource truth may regenerate while Robotino is still beside a
        renewable bank.  During a drained interaction, however, the published
        memory must remain empty so the executor can finish the visit and the
        selector cannot immediately choose regenerated crumbs.  Once Robotino
        leaves beyond ``recharge_rearm_distance``, a later observation may copy
        the regenerated hidden amount again.
        """
        if tag_id not in self.memory:
            return
        resource = self.resource_truth.get(int(tag_id))
        if resource is None:
            return

        if int(tag_id) == int(self.blocked_recharge_target_id):
            self.memory[tag_id]["resource_remaining"] = 0.0
            self.memory[tag_id]["last_supply_reason"] = "empty_current_visit"
            self.memory[tag_id]["last_supply_deliverable"] = 0.0
            return

        self.memory[tag_id]["resource_remaining"] = float(
            resource["remaining"]
        )

    def collect_energy_from_bank(self, tag_id, msg, now_sec):
        if tag_id not in self.memory:
            return 0.0
        if int(tag_id) == int(self.blocked_recharge_target_id):
            return 0.0
        data = self.memory[tag_id]
        resource = self.resource_truth.get(int(tag_id))
        if resource is None or not data.get("is_energy_bank", False):
            return 0.0
        if msg.distance <= 0.05 or msg.distance > self.arrival_distance:
            return 0.0

        remaining = float(resource["remaining"])
        supply = self.evaluate_bank_supply(data, remaining)
        previous_supply_reason = data.get("last_supply_reason", "unverified")
        data["resource_remaining"] = remaining
        data["last_supply_reason"] = supply["reason"]
        data["last_supply_required"] = float(supply["required"])
        data["last_supply_deliverable"] = float(supply["deliverable"])

        if not supply["contains_energy"]:
            if self.active_recharge_target_id == int(tag_id):
                self.active_recharge_target_id = -1
                self.blocked_recharge_target_id = int(tag_id)
                self.get_logger().info(
                    "Recharge interaction exhausted; blocking repeated "
                    "authorization until the executor clears the target: "
                    f"tag_id={int(tag_id)}, "
                    f"remaining={remaining:.3f}."
                )
            return 0.0

        # A bank that is already being used may be drained even when it cannot
        # complete the full recovery. The ranking mixin simultaneously removes
        # it from best-bank selection, so once it reaches the empty epsilon the
        # executor leaves instead of selecting its tiny regeneration again.
        if (
            not supply["actionable"]
            and previous_supply_reason != supply["reason"]
        ):
            self.get_logger().info(
                "Current bank can provide only a partial recovery; draining "
                "remaining observed energy before leaving: "
                f"tag_id={int(tag_id)}, remaining={remaining:.3f}, "
                f"deliverable={supply['deliverable']:.3f}, "
                f"required={supply['required']:.3f}."
            )

        if self.robot_energy >= 1.0:
            return 0.0

        last_time = float(resource.get("last_collection_time", now_sec))
        dt = min(max(0.0, now_sec - last_time), 1.0)
        amount_to_take = float(resource["collection_rate"]) * dt
        amount_taken = min(
            amount_to_take,
            remaining,
            1.0 - self.robot_energy,
        )

        resource["last_collection_time"] = now_sec
        if amount_taken <= 0.0:
            return 0.0

        energy_before = self.robot_energy
        self.robot_energy = self.clamp(
            self.robot_energy + amount_taken,
            0.0,
            1.0,
        )

        new_remaining = max(0.0, remaining - amount_taken)
        drained_this_visit = new_remaining <= self.resource_empty_epsilon
        if drained_this_visit:
            # Clamp immediately, before the next observation can advance
            # regeneration.  This closes the race that previously allowed a
            # renewable bank to regenerate and be consumed continuously while
            # Robotino remained parked beside it.
            new_remaining = 0.0

        resource["remaining"] = new_remaining
        data["resource_remaining"] = new_remaining

        if drained_this_visit:
            data["last_supply_reason"] = "empty_current_visit"
            data["last_supply_deliverable"] = 0.0
            self.active_recharge_target_id = -1
            self.blocked_recharge_target_id = int(tag_id)
            self.get_logger().info(
                "Energy bank drained for the current visit; collection is "
                "latched off until Robotino leaves the bank vicinity: "
                f"tag_id={int(tag_id)}, "
                f"energy={energy_before:.3f}->{self.robot_energy:.3f}, "
                f"rearm_distance={self.recharge_rearm_distance:.2f}m."
            )

        self.get_logger().debug(
            "Authorized energy transfer: "
            f"tag_id={tag_id}, amount={amount_taken:.4f}, "
            f"energy={energy_before:.3f}->{self.robot_energy:.3f}, "
            f"remaining={resource['remaining']:.3f}."
        )
        return float(amount_taken)