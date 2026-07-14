"""RobotinoPolicyOutcome publishing and simple actuator helpers."""

from typing import Optional

from geometry_msgs.msg import PoseStamped, Twist
from robotino_emdb_interfaces.msg import (
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)
from std_msgs.msg import Bool

from . import constants


class OutcomePublishingMixin:
    """Build and publish the compact RobotinoPolicyOutcome interface.

    RobotinoSelectedPolicy policy IDs and RobotinoPolicyOutcome policy IDs are two
    different enums. The helpers below perform the translation explicitly so
    the LTM never has to understand executor-internal policy numbers.
    """

    def publish_simple_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        success: bool,
        failure_reason: str = constants.FAILURE_NONE,
        navigation_result: int = RobotinoPolicyOutcome.NAV_NOT_USED,
        tag_result: int = RobotinoPolicyOutcome.TAG_NOT_CHECKED,
        detection_confidence: Optional[float] = None,
        observed_tag_pose: Optional[PoseStamped] = None,
        recharge_attempted: bool = False,
        recharge_succeeded: bool = False,
        policy_completed: bool = True,
    ) -> None:
        """Publish an outcome that starts and finishes in one callback."""
        self.publish_policy_outcome(
            policy=policy,
            policy_completed=policy_completed,
            policy_success=success,
            failure_reason=failure_reason,
            energy_before=self.get_current_energy(),
            target_type=self.target_type_for_policy(policy),
            target_id=int(policy.target_tag_id),
            navigation_result=navigation_result,
            tag_result=tag_result,
            detection_confidence=detection_confidence,
            observed_tag_pose=observed_tag_pose,
            recharge_attempted=recharge_attempted,
            recharge_succeeded=recharge_succeeded,
        )

    def publish_policy_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        policy_completed: bool,
        policy_success: bool,
        failure_reason: str,
        energy_before: float,
        target_type: Optional[int] = None,
        target_id: Optional[int] = None,
        navigation_result: int = RobotinoPolicyOutcome.NAV_NOT_USED,
        tag_result: int = RobotinoPolicyOutcome.TAG_NOT_CHECKED,
        detection_confidence: Optional[float] = None,
        observed_tag_pose: Optional[PoseStamped] = None,
        recharge_attempted: bool = False,
        recharge_succeeded: bool = False,
    ) -> None:
        """Publish one objective execution result for the LTM."""
        outcome = RobotinoPolicyOutcome()
        outcome.header.stamp = self.get_clock().now().to_msg()
        outcome.header.frame_id = self.map_frame

        outcome.policy_id = int(self.outcome_policy_id_for_policy(policy))
        outcome.policy_completed = bool(policy_completed)
        outcome.policy_success = bool(policy_success)

        if target_type is None:
            target_type = self.target_type_for_policy(policy)
        outcome.target_type = int(target_type)

        resolved_target_id = (
            int(policy.target_tag_id)
            if target_id is None
            else int(target_id)
        )
        outcome.target_id = resolved_target_id

        outcome.navigation_result = int(navigation_result)
        outcome.tag_result = int(tag_result)
        outcome.detection_confidence = float(
            0.0 if detection_confidence is None else detection_confidence
        )

        outcome.observed_tag_pose_valid = observed_tag_pose is not None
        if observed_tag_pose is not None:
            outcome.observed_tag_pose = observed_tag_pose

        outcome.recharge_attempted = bool(recharge_attempted)
        outcome.recharge_succeeded = bool(recharge_succeeded)

        outcome.energy_before = float(energy_before)
        outcome.energy_after = float(self.get_current_energy())

        # A successful or still-running policy has no failure.
        if policy_success or not policy_completed:
            outcome.failure_reason = constants.FAILURE_NONE
        else:
            resolved_reason = str(failure_reason or constants.FAILURE_UNKNOWN)
            if resolved_reason not in constants.ALL_FAILURE_REASONS:
                self.get_logger().warn(
                    "Publishing unregistered RobotinoPolicyOutcome failure reason: "
                    f"'{resolved_reason}'."
                )
            outcome.failure_reason = resolved_reason

        self.outcome_publisher.publish(outcome)

    def outcome_policy_id_for_policy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> int:
        """Translate executor policy IDs into RobotinoPolicyOutcome policy IDs."""
        policy_id = int(policy.policy_id)

        if policy_id in (
            self.POLICY_CONTINUE_EXPLORING,
            self.POLICY_SEARCH_FOR_ENERGY,
        ):
            return RobotinoPolicyOutcome.POLICY_EXPLORE

        if policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            if (
                self.visible_tag_target_type(policy)
                == RobotinoPolicyOutcome.TARGET_ENERGY_TAG
            ):
                return RobotinoPolicyOutcome.POLICY_VERIFY_ENERGY
            # RobotinoPolicyOutcome.msg has no generic semantic-inspection policy.
            return RobotinoPolicyOutcome.POLICY_UNKNOWN

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            return RobotinoPolicyOutcome.POLICY_RETURN_TO_ENERGY

        if policy_id == self.POLICY_GOAL:
            return RobotinoPolicyOutcome.POLICY_GO_TO_GOAL

        return RobotinoPolicyOutcome.POLICY_UNKNOWN

    def target_type_for_policy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> int:
        policy_id = int(policy.policy_id)

        if policy_id == self.POLICY_RETURN_TO_BEST_ENERGY_BANK:
            return RobotinoPolicyOutcome.TARGET_ENERGY_TAG

        if policy_id == self.POLICY_GOAL:
            return RobotinoPolicyOutcome.TARGET_GOAL_TAG

        if policy_id == self.POLICY_INSPECT_VISIBLE_TAG:
            return self.visible_tag_target_type(policy)

        return RobotinoPolicyOutcome.TARGET_NONE

    def visible_tag_target_type(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> int:
        """Infer a visible tag's semantic type without inventing one."""
        state = self.latest_foraging_state
        if state is None:
            return RobotinoPolicyOutcome.TARGET_NONE

        state_tag_id = int(getattr(state, "tag_id", -1))
        if state_tag_id != int(policy.target_tag_id):
            return RobotinoPolicyOutcome.TARGET_NONE

        if bool(getattr(state, "is_energy_bank", False)):
            return RobotinoPolicyOutcome.TARGET_ENERGY_TAG

        tag_type = str(getattr(state, "tag_type", "")).strip().lower()
        if "energy" in tag_type:
            return RobotinoPolicyOutcome.TARGET_ENERGY_TAG
        if "goal" in tag_type:
            return RobotinoPolicyOutcome.TARGET_GOAL_TAG

        return RobotinoPolicyOutcome.TARGET_NONE

    def detection_confidence_for_policy(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> float:
        state = self.latest_foraging_state
        if state is None:
            return 0.0

        state_tag_id = int(getattr(state, "tag_id", -1))
        if state_tag_id != int(policy.target_tag_id):
            return 0.0

        return max(0.0, float(getattr(state, "confidence", 0.0)))

    def get_current_energy(self) -> float:
        if self.latest_foraging_state is None:
            return 0.0
        return float(self.latest_foraging_state.robot_energy)

    def get_current_reward(self) -> float:
        if self.latest_foraging_state is None:
            return 0.0
        return float(self.latest_foraging_state.total_reward)

    def set_exploration_enabled(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.exploration_enable_publisher.publish(msg)

    def stop_robot(self) -> None:
        self.cmd_vel_publisher.publish(Twist())

    def resume_after_failed_execution_if_configured(self) -> None:
        if self.resume_exploration_after_failure:
            self.set_exploration_enabled(True)
