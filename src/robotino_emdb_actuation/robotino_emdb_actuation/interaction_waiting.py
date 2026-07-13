"""Semantic interaction confirmation.

After Nav2 reaches a tag's standoff pose, this module waits for a real
state change (energy increase for the energy bank, reward/goal-flag
change for the goal) before reporting semantic success. Reaching a pose
alone is never treated as success.
"""


class InteractionWaitingMixin:
    """Requires from the host class:

    Attributes: latest_foraging_state, interaction_timeout_s,
        energy_success_delta, reward_success_delta.
    Methods: get_execution(), get_waiting_context(), get_current_energy(),
        get_current_reward(), publish_policy_outcome(), finish_execution().
    Constants: STAGE_NAVIGATING, STAGE_WAITING_INTERACTION,
        POLICY_RETURN_TO_BEST_ENERGY_BANK, POLICY_GOAL.
    """

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
            status = (
                "energy_approach_pose_reached_waiting_for_charge_"
                f"{candidate_name}"
            )
        elif policy_id == self.POLICY_GOAL:
            status = (
                "goal_approach_pose_reached_waiting_for_reward_"
                f"{candidate_name}"
            )
        else:
            # Only energy and goal policies should use tag navigation now.
            self.finish_execution(
                generation,
                success=True,
                status=f"tag_approach_pose_reached_{candidate_name}",
                resume_exploration=False,
            )
            return

        self.get_logger().info(status)

        # This is intentionally unfinished and unsuccessful. Reaching a Nav2
        # standoff pose is not the same as charging or completing the goal.
        self.publish_policy_outcome(
            policy,
            started=True,
            finished=False,
            success=False,
            status=status,
            energy_before=float(context["energy_before"]),
            reward_before=float(context["reward_before"]),
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
                # Establish a valid baseline on the first state sample instead
                # of comparing a real energy value against the fallback 0.0.
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
                    status="energy_charge_confirmed",
                    resume_exploration=self.resume_exploration_after_energy,
                )

        elif policy_id == self.POLICY_GOAL:
            if context["arrival_reward"] is None:
                # Establish both baselines on the first state sample. This
                # prevents a late first state message from looking like reward.
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

            if (
                reward_delta >= self.reward_success_delta
                or goal_flag_transition
            ):
                self.get_logger().info(
                    "Goal interaction confirmed: "
                    f"reward increased by {reward_delta:.3f}."
                )
                self.finish_execution(
                    generation,
                    success=True,
                    status="goal_interaction_confirmed",
                    resume_exploration=False,
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
            status = "energy_interaction_timeout_no_charge_detected"
        elif policy_id == self.POLICY_GOAL:
            status = "goal_interaction_timeout_no_reward_detected"
        else:
            status = "interaction_timeout"

        self.get_logger().warn(status)
        self.finish_execution(
            generation,
            success=False,
            status=status,
            resume_exploration=self.resume_exploration_after_failure,
        )

    def foraging_state_reports_goal_complete(self) -> bool:
        """Use an explicit completion flag when the message provides one.

        The currently used message is known to contain total_reward. This
        introspection also supports future interface versions without making
        this node depend on a field that may not yet exist.
        """
        if self.latest_foraging_state is None:
            return False

        for field_name in (
            "goal_reached",
            "goal_achieved",
            "task_complete",
            "task_completed",
        ):
            if hasattr(self.latest_foraging_state, field_name):
                return bool(getattr(self.latest_foraging_state, field_name))

        return False