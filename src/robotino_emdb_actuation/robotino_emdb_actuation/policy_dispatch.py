"""Inputs and high-level policy dispatch."""

from robotino_emdb_interfaces.msg import (
    RobotinoPolicyOutcome,
    RobotinoForagingState,
    RobotinoSelectedPolicy,
)

from . import constants


class PolicyDispatchMixin:
    """Receive eMDB-selected policies and route them to execution paths."""

    EXPLORATION_MODE_NOVELTY = "novelty"
    EXPLORATION_MODE_ENERGY_SEARCH = "energy_search"

    def update_energy_recovery_mode(self, energy: float) -> None:
        """Keep actuator safety synchronized with context hysteresis."""
        energy = max(0.0, min(1.0, float(energy)))

        if not self.energy_mode_initialized:
            self.energy_recovery_mode = (
                energy <= self.wander_energy_threshold
            )
            self.energy_mode_initialized = True
            return

        if (
            not self.energy_recovery_mode
            and energy <= self.wander_energy_threshold
        ):
            self.energy_recovery_mode = True
            self.get_logger().info(
                f"Executor entered energy recovery at energy={energy:.3f}."
            )
        elif (
            self.energy_recovery_mode
            and energy >= self.resume_energy_threshold
        ):
            self.energy_recovery_mode = False
            self.get_logger().info(
                f"Executor left energy recovery at energy={energy:.3f}."
            )

    def frontier_exploration_allowed(self, mode: str, state) -> bool:
        """Return whether the persistent frontier controller may move."""
        if state is None or not bool(getattr(state, "valid", False)):
            return False

        if self.mapping_complete:
            return False

        if bool(getattr(state, "goal_satisfied", False)):
            return False

        if mode == self.EXPLORATION_MODE_ENERGY_SEARCH:
            return (
                self.energy_recovery_mode
                and not self.energy_bank_is_worthy(state)
            )

        return not self.energy_recovery_mode

    def enforce_frontier_exploration_safety(self, state) -> None:
        """Revoke persistent frontier motion when its context expires."""
        if not self.frontier_exploration_enabled:
            return

        mode = (
            self.frontier_exploration_mode
            or self.EXPLORATION_MODE_NOVELTY
        )
        if self.frontier_exploration_allowed(mode, state):
            return

        self.get_logger().warn(
            "Disabling persistent frontier exploration because its "
            f"'{mode}' context is no longer valid: "
            f"energy={float(getattr(state, 'robot_energy', 0.0)):.3f}, "
            f"recovery={self.energy_recovery_mode}, "
            f"mapping_complete={self.mapping_complete}."
        )
        self.set_exploration_enabled(False)

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        self.latest_foraging_state = msg
        self.update_energy_recovery_mode(float(msg.robot_energy))
        self.enforce_frontier_exploration_safety(msg)
        self.handle_wander_state_change(msg)

        if self.is_waiting_for_interaction():
            self.check_interaction_completion()

    def policy_callback(self, msg: RobotinoSelectedPolicy) -> None:
        if not bool(msg.valid) or not bool(msg.execute_now):
            return

        policy_id = int(msg.policy_id)

        if policy_id == self.POLICY_CONTINUE_EXPLORING:
            if str(msg.policy_name).strip().lower() == "wander_mapped_space":
                self.execute_wander_mapped_space(msg)
            else:
                self.execute_continue_exploring(msg)
        elif policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            self.execute_inspect_visible_tag(msg)
        elif policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.execute_return_to_best_energy_bank(msg)
        elif policy_id == self.POLICY_SEARCH_FOR_ENERGY:
            if bool(msg.use_nav2):
                self.execute_search_for_energy_mapped_space(msg)
            else:
                self.execute_search_for_energy(msg)
        elif policy_id == self.POLICY_GOAL:
            self.execute_goal(msg)
        else:
            self.get_logger().warn(
                f"Unsupported policy_id={policy_id}, "
                f"policy_name='{msg.policy_name}'."
            )
            self.publish_simple_outcome(
                msg,
                success=False,
                failure_reason=constants.FAILURE_POLICY_NOT_SUPPORTED,
            )

    def execute_continue_exploring(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_continue_exploring",
            publish_outcome=True,
        )

        if not self.frontier_exploration_allowed(
            self.EXPLORATION_MODE_NOVELTY,
            self.latest_foraging_state,
        ):
            self.get_logger().warn(
                "Rejected continue_exploring because energy recovery, "
                "mapping completion, or mission completion has priority."
            )
            self.set_exploration_enabled(False)
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
            )
            return

        self.set_exploration_enabled(
            True,
            mode=self.EXPLORATION_MODE_NOVELTY,
        )
        self.publish_simple_outcome(
            policy,
            success=True,
            failure_reason=constants.FAILURE_NONE,
        )

    def execute_search_for_energy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_search_for_energy",
            publish_outcome=True,
        )

        if not self.frontier_exploration_allowed(
            self.EXPLORATION_MODE_ENERGY_SEARCH,
            self.latest_foraging_state,
        ):
            self.get_logger().warn(
                "Rejected frontier search_for_energy because recovery is "
                "inactive, mapping is complete, or a worthy bank is known."
            )
            self.set_exploration_enabled(False)
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
            )
            return

        self.set_exploration_enabled(
            True,
            mode=self.EXPLORATION_MODE_ENERGY_SEARCH,
        )
        self.publish_simple_outcome(
            policy,
            success=True,
            failure_reason=constants.FAILURE_NONE,
        )

    def execute_inspect_visible_tag(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        """Acknowledge a visible tag without changing robot motion."""
        active_description = self.describe_active_execution()
        confidence = self.detection_confidence_for_policy(policy)

        self.get_logger().info(
            f"Observed tag {int(policy.target_tag_id)} saved by memory layer; "
            f"motion unchanged ({active_description})."
        )
        self.publish_simple_outcome(
            policy,
            success=True,
            failure_reason=constants.FAILURE_NONE,
            navigation_result=RobotinoPolicyOutcome.NAV_NOT_USED,
            tag_result=RobotinoPolicyOutcome.TAG_FOUND,
            detection_confidence=confidence,
        )

    def execute_return_to_best_energy_bank(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated energy-return policy ignored; execution is active."
            )
            return

        # Energy has priority over goal.
        if self.execution is not None:
            self.cancel_current_execution(
                reason="preempted_by_higher_priority_energy_policy",
                publish_outcome=True,
            )

        self.start_tag_navigation(
            policy,
            approach_standoff_distance=self.energy_approach_standoff_m,
        )

    def execute_goal(self, policy: RobotinoSelectedPolicy) -> None:
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated goal policy ignored; execution is already active."
            )
            return

        # A goal may not interrupt returning to an energy bank.
        if self.active_policy_id() == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.get_logger().warn(
                "Goal policy deferred because an energy-return policy is "
                "active. The goal remains stored for later."
            )
            return

        if self.execution is not None:
            self.cancel_current_execution(
                reason="preempted_by_new_goal_policy",
                publish_outcome=True,
            )

        self.start_tag_navigation(
            policy,
            approach_standoff_distance=self.goal_approach_standoff_m,
        )
