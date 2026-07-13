"""Execution lifecycle and stale-callback safety.

Every in-flight execution is tagged with a generation token. Incrementing
the token invalidates any callback still in flight from an old or
cancelled execution, so a late Nav2/ComputePathToPose result can never be
attributed to the wrong policy.
"""

import math
from typing import Any, Dict, Optional

from robotino_emdb_interfaces.msg import RobotinoSelectedPolicy


class ExecutionLifecycleMixin:
    """Requires from the host class:

    Attributes: execution, execution_generation, active_path_goal_handle,
        active_navigation_goal_handle, candidate_plan_candidates,
        candidate_plan_index.
    Methods: publish_policy_outcome(), set_exploration_enabled(),
        stop_robot().
    Constants: STAGE_WAITING_INTERACTION.
    """

    def finish_execution(
        self,
        generation: int,
        success: bool,
        status: str,
        resume_exploration: bool,
    ) -> None:
        context = self.get_execution(generation)
        if context is None:
            return

        policy = context["policy"]
        energy_before = float(context["energy_before"])
        reward_before = float(context["reward_before"])

        # Invalidate callbacks before publishing or enabling another behavior.
        self.execution_generation += 1
        self.execution = None
        self.active_path_goal_handle = None
        self.active_navigation_goal_handle = None
        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0

        self.publish_policy_outcome(
            policy,
            started=True,
            finished=True,
            success=success,
            status=status,
            energy_before=energy_before,
            reward_before=reward_before,
        )

        if resume_exploration:
            self.set_exploration_enabled(True)
            self.get_logger().info(
                "Exploration re-enabled after policy completion."
            )

    def cancel_current_execution(
        self,
        reason: str,
        publish_outcome: bool,
    ) -> None:
        old_context = self.execution

        # Invalidate all in-flight callbacks first.
        self.execution_generation += 1
        self.execution = None

        if self.active_path_goal_handle is not None:
            try:
                self.active_path_goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    f"Failed to cancel path-planning goal: {ex}"
                )
            self.active_path_goal_handle = None

        if self.active_navigation_goal_handle is not None:
            try:
                self.active_navigation_goal_handle.cancel_goal_async()
            except Exception as ex:  # noqa: BLE001
                self.get_logger().warn(
                    f"Failed to cancel navigation goal: {ex}"
                )
            self.active_navigation_goal_handle = None

        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0

        if old_context is None:
            return

        self.stop_robot()
        old_policy = old_context["policy"]
        self.get_logger().info(
            f"Cancelled policy {int(old_policy.policy_id)} "
            f"'{old_policy.policy_name}': {reason}."
        )

        if publish_outcome:
            self.publish_policy_outcome(
                old_policy,
                started=True,
                finished=True,
                success=False,
                status=reason,
                energy_before=float(old_context["energy_before"]),
                reward_before=float(old_context["reward_before"]),
            )

    def get_execution(
        self,
        generation: int,
        required_stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.execution is None:
            return None

        if int(self.execution["generation"]) != int(generation):
            return None

        if required_stage is not None:
            if self.execution.get("stage") != required_stage:
                return None

        return self.execution

    def get_waiting_context(self) -> Optional[Dict[str, Any]]:
        if self.execution is None:
            return None
        if self.execution.get("stage") != self.STAGE_WAITING_INTERACTION:
            return None
        return self.execution

    def is_waiting_for_interaction(self) -> bool:
        return self.get_waiting_context() is not None

    def active_policy_id(self) -> Optional[int]:
        if self.execution is None:
            return None
        return int(self.execution["policy"].policy_id)

    def is_same_policy_already_active(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> bool:
        if self.execution is None:
            return False

        active_policy = self.execution["policy"]
        return (
            int(active_policy.policy_id) == int(policy.policy_id)
            and int(active_policy.target_tag_id) == int(policy.target_tag_id)
            and math.isclose(
                float(active_policy.target_x_map),
                float(policy.target_x_map),
                abs_tol=0.02,
            )
            and math.isclose(
                float(active_policy.target_y_map),
                float(policy.target_y_map),
                abs_tol=0.02,
            )
        )

    def describe_active_execution(self) -> str:
        if self.execution is None:
            return "no active navigation policy"

        policy = self.execution["policy"]
        return (
            f"policy_id={int(policy.policy_id)}, "
            f"stage={self.execution.get('stage', 'unknown')}"
        )