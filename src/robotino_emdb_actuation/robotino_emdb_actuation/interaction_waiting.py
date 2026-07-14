"""Semantic interaction confirmation.

After Nav2 reaches a tag's standoff pose, this module waits for a real state
change before reporting semantic success. Reaching a pose alone is not treated
as success.
"""

from robotino_emdb_interfaces.msg import RobotinoPolicyOutcome

from . import constants


class InteractionWaitingMixin:
    """Post-navigation energy/reward confirmation."""

    def enter_interaction_wait(self, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            return

        policy = context["policy"]
        policy_id = int(policy.policy_id)

        context["stage"] = self.STAGE_WAITING_INTERACTION
        if self.latest_foraging_state is None:
            context["arrival_energy"] = None
            context["arrival_reward"] = None
            context["arrival_goal_complete"] = False
        else:
            context["arrival_energy"] = self.get_current_energy()
            context["arrival_reward"] = self.get_current_reward()
            context["arrival_goal_complete"] = (
                self.foraging_state_reports_goal_complete()
            )

        context["interaction_deadline_ns"] = (
            self.get_clock().now().nanoseconds
            + int(self.interaction_timeout_s * 1e9)
        )

        candidate_name = context.get("selected_candidate_name") or "candidate"

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            event_description = (
                "energy_approach_pose_reached_waiting_for_charge_"
                f"{candidate_name}"
            )
        elif policy_id == self.POLICY_GOAL:
            event_description = (
                "goal_approach_pose_reached_waiting_for_reward_"
                f"{candidate_name}"
            )
        else:
            # Only energy and goal policies should use tag navigation here.
            self.finish_execution(
                generation,
                success=True,
                failure_reason=constants.FAILURE_NONE,
                resume_exploration=False,
                navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                tag_result=RobotinoPolicyOutcome.TAG_NOT_CHECKED,
            )
            return

        self.get_logger().info(event_description)

        # In-progress notification. Nav2 succeeded, but the semantic policy is
        # not complete until charging/reward is confirmed.
        self.publish_policy_outcome(
            policy=policy,
            policy_completed=False,
            policy_success=False,
            failure_reason=constants.FAILURE_NONE,
            energy_before=float(context["energy_before"]),
            target_type=self.target_type_for_policy(policy),
            target_id=int(policy.target_tag_id),
            navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
            tag_result=RobotinoPolicyOutcome.TAG_NOT_CHECKED,
            detection_confidence=None,
            observed_tag_pose=None,
            recharge_attempted=(
                policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK
            ),
            recharge_succeeded=False,
        )

        self.check_interaction_completion()

    def check_interaction_completion(self) -> None:
        context = self.get_waiting_context()
        if context is None:
            return

        policy = context["policy"]
        policy_id = int(policy.policy_id)
        generation = int(context["generation"])

        if self.latest_foraging_state is None:
            return

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            if context["arrival_energy"] is None:
                # Establish a real baseline on the first state sample.
                context["arrival_energy"] = self.get_current_energy()
                return

            arrival_energy = float(context["arrival_energy"])
            current_energy = self.get_current_energy()
            energy_delta = current_energy - arrival_energy

            if energy_delta >= self.energy_success_delta:
                self.get_logger().info(
                    "Charging confirmed: "
                    f"energy increased by {energy_delta:.3f}."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    failure_reason=constants.FAILURE_NONE,
                    resume_exploration=self.resume_exploration_after_energy,
                    navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                    tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                    recharge_attempted=True,
                    recharge_succeeded=True,
                )

        elif policy_id == self.POLICY_GOAL:
            if context["arrival_reward"] is None:
                context["arrival_reward"] = self.get_current_reward()
                context["arrival_goal_complete"] = (
                    self.foraging_state_reports_goal_complete()
                )
                return

            arrival_reward = float(context["arrival_reward"])
            current_reward = self.get_current_reward()
            reward_delta = current_reward - arrival_reward
            goal_flag_transition = (
                not bool(context.get("arrival_goal_complete", False))
                and self.foraging_state_reports_goal_complete()
            )

            if reward_delta >= self.reward_success_delta or goal_flag_transition:
                self.get_logger().info(
                    "Goal interaction confirmed: "
                    f"reward increased by {reward_delta:.3f}."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    failure_reason=constants.FAILURE_NONE,
                    resume_exploration=False,
                    navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
                    tag_result=RobotinoPolicyOutcome.TAG_FOUND,
                )

    def interaction_timer_callback(self) -> None:
        context = self.get_waiting_context()
        if context is None:
            return

        self.check_interaction_completion()

        # check_interaction_completion() may have finished and cleared it.
        context = self.get_waiting_context()
        if context is None:
            return

        deadline_ns = context.get("interaction_deadline_ns")
        if deadline_ns is None:
            return

        if self.get_clock().now().nanoseconds < int(deadline_ns):
            return

        policy_id = int(context["policy"].policy_id)
        generation = int(context["generation"])

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            failure_reason = constants.FAILURE_RECHARGE_TIMEOUT
            event_description = "energy_interaction_timeout_no_charge_detected"
        elif policy_id == self.POLICY_GOAL:
            failure_reason = constants.FAILURE_GOAL_NOT_CONFIRMED
            event_description = "goal_interaction_timeout_no_reward_detected"
        else:
            failure_reason = constants.FAILURE_OBSERVATION_TIMEOUT
            event_description = "interaction_timeout"

        self.get_logger().warn(event_description)
        self.finish_execution(
            generation,
            success=False,
            failure_reason=failure_reason,
            resume_exploration=self.resume_exploration_after_failure,
            navigation_result=RobotinoPolicyOutcome.NAV_SUCCEEDED,
            tag_result=RobotinoPolicyOutcome.TAG_CHECK_INCONCLUSIVE,
            recharge_attempted=(
                policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK
            ),
            recharge_succeeded=False,
        )

    def foraging_state_reports_goal_complete(self) -> bool:
        """Use an explicit completion flag when the message provides one."""
        if self.latest_foraging_state is None:
            return False

        for field_name in (
            "goal_reached",
            "goal_achieved",
            "goal_satisfied",
            "task_complete",
            "task_completed",
        ):
            if hasattr(self.latest_foraging_state, field_name):
                return bool(getattr(self.latest_foraging_state, field_name))

        return False
