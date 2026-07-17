"""Execution lifecycle and stale-callback safety.

Every in-flight execution is tagged with a generation token. Incrementing
the token invalidates callbacks still in flight from an old or cancelled
execution, so late Nav2/ComputePathToPose results cannot be attributed to a
newer policy.
"""

from typing import Any, Dict, Optional

from geometry_msgs.msg import PoseStamped
from robotino_emdb_interfaces.msg import RobotinoPolicyOutcome, RobotinoSelectedPolicy

from . import constants


class ExecutionLifecycleMixin:
    """Shared completion, cancellation, and active-execution helpers."""

    def finish_execution(
        self,
        generation: int,
        success: bool,
        failure_reason: str,
        resume_exploration: bool,
        navigation_result: int = RobotinoPolicyOutcome.NAV_NOT_USED,
        tag_result: int = RobotinoPolicyOutcome.TAG_NOT_CHECKED,
        detection_confidence: Optional[float] = None,
        observed_tag_pose: Optional[PoseStamped] = None,
        recharge_attempted: bool = False,
        recharge_succeeded: bool = False,
    ) -> None:
        """Close the current execution and publish its final outcome."""
        context = self.get_execution(generation)
        if context is None:
            return

        policy = context["policy"]
        was_wandering = context.get("navigation_mode") == "wander"
        energy_before = float(context["energy_before"])
        target_type = self.target_type_for_policy(policy)
        target_id = int(policy.target_tag_id)

        # Invalidate callbacks before publishing or enabling another behavior.
        self.execution_generation += 1
        self.execution = None
        self.active_path_goal_handle = None
        self.active_navigation_goal_handle = None
        self.candidate_plan_candidates = []
        self.candidate_plan_index = 0
        if was_wandering:
            self.set_wandering_active(False)

        self.publish_policy_outcome(
            policy=policy,
            policy_completed=True,
            policy_success=success,
            failure_reason=failure_reason,
            energy_before=energy_before,
            target_type=target_type,
            target_id=target_id,
            navigation_result=navigation_result,
            tag_result=tag_result,
            detection_confidence=detection_confidence,
            observed_tag_pose=observed_tag_pose,
            recharge_attempted=recharge_attempted,
            recharge_succeeded=recharge_succeeded,
        )

        if resume_exploration:
            self.resume_frontier_exploration_if_allowed()
        else:
            self.set_exploration_enabled(False)

    def cancel_current_execution(
        self,
        reason: str,
        publish_outcome: bool,
    ) -> None:
        old_context = self.execution
        was_wandering = (
            old_context is not None
            and old_context.get("navigation_mode") == "wander"
        )

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
        if was_wandering:
            self.set_wandering_active(False)

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
                policy=old_policy,
                policy_completed=True,
                policy_success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
                energy_before=float(old_context["energy_before"]),
                target_type=self.target_type_for_policy(old_policy),
                target_id=int(old_policy.target_tag_id),
                navigation_result=RobotinoPolicyOutcome.NAV_CANCELED,
                tag_result=RobotinoPolicyOutcome.TAG_NOT_CHECKED,
                detection_confidence=None,
                observed_tag_pose=None,
                recharge_attempted=False,
                recharge_succeeded=False,
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
        """Return whether an incoming command matches the active execution.

        Tag coordinates can move by several centimeters as AprilTag perception
        and memory estimates are updated. Once navigation toward a semantic tag
        is active, coordinate jitter must not cancel and restart the Nav2 goal.

        For tag-navigation policies, identity is therefore the policy ID plus
        target tag ID. For policies without a semantic tag target, matching the
        policy ID is sufficient to suppress repeated publications while the
        current execution remains active.
        """
        if self.execution is None:
            return False

        active_policy = self.execution.get("policy")
        if active_policy is None:
            return False

        active_policy_id = int(active_policy.policy_id)
        incoming_policy_id = int(policy.policy_id)
        if active_policy_id != incoming_policy_id:
            return False

        tag_navigation_policies = {
            self.POLICY_RETURN_TO_BEST_ENERGY_BANK,
            self.POLICY_GOAL,
        }
        if incoming_policy_id in tag_navigation_policies:
            return (
                int(active_policy.target_tag_id)
                == int(policy.target_tag_id)
            )

        return True

    def describe_active_execution(self) -> str:
        if self.execution is None:
            return "no active navigation policy"

        policy = self.execution["policy"]
        return (
            f"policy_id={int(policy.policy_id)}, "
            f"stage={self.execution.get('stage', 'unknown')}"
        )