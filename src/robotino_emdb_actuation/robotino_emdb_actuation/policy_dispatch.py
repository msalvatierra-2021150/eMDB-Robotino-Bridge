"""Inputs and high-level policy dispatch.

Receives the eMDB-selected policy and foraging state, and routes each
selected policy to the correct execution path.
"""

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoSelectedPolicy,
)


class PolicyDispatchMixin:
    """Requires from the host class:

    Attributes: latest_foraging_state, execution.
    Methods: is_waiting_for_interaction(), check_interaction_completion(),
        execute_continue_exploring(), execute_inspect_visible_tag(),
        execute_return_to_best_energy_bank(), execute_search_for_energy(),
        execute_goal(), cancel_current_execution(), start_tag_navigation(),
        publish_simple_outcome(), set_exploration_enabled(),
        is_same_policy_already_active(), active_policy_id(),
        describe_active_execution().
    Constants: POLICY_CONTINUE_EXPLORING, POLICY_INSPECT_VISIBLE_TAG,
        POLICY_RETURN_TO_BEST_ENERGY_BANK, POLICY_SEARCH_FOR_ENERGY,
        POLICY_GOAL.
    """

    def foraging_callback(self, msg: RobotinoForagingState) -> None:
        self.latest_foraging_state = msg

        if self.is_waiting_for_interaction():
            self.check_interaction_completion()

    def policy_callback(self, msg: RobotinoSelectedPolicy) -> None:
        if not bool(msg.valid) or not bool(msg.execute_now):
            return

        policy_id = int(msg.policy_id)

        if policy_id == self.POLICY_CONTINUE_EXPLORING:
            self.execute_continue_exploring(msg)
        elif policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            self.execute_inspect_visible_tag(msg)
        elif policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.execute_return_to_best_energy_bank(msg)
        elif policy_id == self.POLICY_SEARCH_FOR_ENERGY:
            self.execute_search_for_energy(msg)
        elif policy_id == self.POLICY_GOAL:
            self.execute_goal(msg)
        else:
            self.get_logger().warn(
                f"Ignoring unknown policy_id={policy_id}, "
                f"policy_name='{msg.policy_name}'."
            )

    def execute_continue_exploring(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_continue_exploring",
            publish_outcome=True,
        )
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="continuing_exploration",
        )

    def execute_search_for_energy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        self.cancel_current_execution(
            reason="preempted_by_search_for_energy",
            publish_outcome=True,
        )
        self.set_exploration_enabled(True)
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="searching_for_energy",
        )

    def execute_inspect_visible_tag(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        """Acknowledge perception/memory without changing robot motion.

        The perception/memory layer is responsible for saving the tag. This
        executor intentionally does not cancel exploration, cancel an energy
        return, disable exploration, or issue a Nav2 goal here.
        """
        active_description = self.describe_active_execution()
        self.get_logger().info(
            f"Observed tag {int(policy.target_tag_id)} saved by memory layer; "
            f"motion unchanged ({active_description})."
        )
        self.publish_simple_outcome(
            policy,
            started=True,
            success=True,
            status="tag_observed_and_saved_no_motion",
        )

    def execute_return_to_best_energy_bank(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated energy-return policy ignored; execution is already active."
            )
            return

        # Energy has priority over goal. If a goal approach is running, close
        # it as preempted and begin the energy return immediately.
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

        # Defensive priority guard. A goal policy may not interrupt returning
        # to an energy bank or waiting for charging confirmation.
        if self.active_policy_id() == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            self.get_logger().warn(
                "Goal policy deferred because an energy-return policy is active. "
                "The goal remains stored for later."
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