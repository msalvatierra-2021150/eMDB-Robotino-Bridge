"""Policy-outcome interpretation and per-tag evidence updates."""

from robotino_emdb_interfaces.msg import RobotinoPolicyOutcome

class PolicyOutcomeMixin:
    """Mixin extracted from RobotinoForagingMemory."""

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

        if msg.policy_completed and msg.recharge_attempted:
            data["last_recharge_attempt_time"] = (
                self.get_clock().now().nanoseconds / 1e9
            )

        if self.is_complete_successful_recharge(msg):
            self.handle_successful_recharge(data)
        elif self.is_insufficient_supply_outcome(msg, data):
            self.handle_insufficient_supply(data, msg)
        elif self.is_complete_failed_recharge(msg):
            self.handle_failed_recharge(data)
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
        )

    def is_insufficient_supply_outcome(self, msg, data):
        """Return true when the bank exists but cannot finish recovery.

        This is not a recharge-reliability failure. The resource may have
        transferred energy successfully; it simply lacks enough observed net
        supply to reach the recovery-release threshold.
        """
        if not (
            msg.policy_id
            == RobotinoPolicyOutcome.POLICY_RETURN_TO_ENERGY
            and msg.policy_completed
            and msg.navigation_result
            == RobotinoPolicyOutcome.NAV_SUCCEEDED
            and msg.tag_result == RobotinoPolicyOutcome.TAG_FOUND
        ):
            return False

        supply = self.evaluate_bank_supply(
            data,
            data.get("resource_remaining", 0.0),
        )
        return supply["reason"] == "insufficient_for_recovery"

    @staticmethod
    def is_complete_failed_recharge(msg):
        """A reached resource produced no confirmed energy increase."""
        return (
            msg.policy_id
            == RobotinoPolicyOutcome.POLICY_RETURN_TO_ENERGY
            and msg.policy_completed
            and msg.navigation_result
            == RobotinoPolicyOutcome.NAV_SUCCEEDED
            and msg.recharge_attempted
            and not msg.recharge_succeeded
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
        data["consecutive_recharge_failures"] = 0
        data["status"] = "ACTIVE"
        data["last_outcome"] = "SUCCESSFUL_RECHARGE"
        data["last_failure_reason"] = "none"

    def handle_insufficient_supply(self, data, msg):
        """Record a reachable working bank that cannot complete recovery."""
        data["navigation_attempts"] += 1
        data["navigation_successes"] += 1
        data["verification_attempts"] += 1
        data["presence_positive"] += 0.25
        data["reachability_positive"] += 0.25

        if msg.recharge_attempted:
            data["recharge_attempts"] += 1
        if msg.recharge_succeeded:
            # Partial energy gain proves that the bank works. Do not punish
            # reliability merely because its remaining amount was inadequate.
            data["recharge_successes"] += 1
            data["recharge_positive"] += self.successful_recharge_weight

        data["consecutive_navigation_failures"] = 0
        data["consecutive_not_found"] = 0
        data["consecutive_recharge_failures"] = 0
        data["status"] = "INSUFFICIENT_FOR_RECOVERY"
        data["last_outcome"] = (
            "PARTIAL_RECHARGE_INSUFFICIENT_SUPPLY"
            if msg.recharge_succeeded
            else "INSUFFICIENT_SUPPLY"
        )

    def handle_failed_recharge(self, data):
        """Lower learned recharge reliability without questioning the tag.

        Navigation succeeded and an interaction was attempted, so presence and
        reachability remain supported. Only rechargability receives negative
        evidence. The resource stays remembered as a low-priority future
        verification candidate.
        """
        data["navigation_attempts"] += 1
        data["navigation_successes"] += 1
        data["verification_attempts"] += 1
        data["recharge_attempts"] += 1
        data["recharge_negative"] += self.failed_recharge_weight
        # The latest interaction is stronger evidence than a stale previous
        # observation that the resource was available.
        data["resource_remaining"] = 0.0
        data["consecutive_navigation_failures"] = 0
        data["consecutive_not_found"] = 0
        data["consecutive_recharge_failures"] += 1
        data["status"] = "DEPLETED_OR_NOT_READY"
        data["last_outcome"] = "RECHARGE_FAILED"

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
