"""Output helpers: publishing policy outcomes, reading foraging state, and
simple actuator commands (exploration enable, stop)."""

from robotino_emdb_interfaces.msg import (
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


class OutcomePublishingMixin:
    """Requires from the host class:

    Attributes: latest_foraging_state, outcome_publisher,
        exploration_enable_publisher, cmd_vel_publisher, map_frame,
        resume_exploration_after_failure.
    """

    def publish_simple_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        started: bool,
        success: bool,
        status: str,
    ) -> None:
        energy_before = self.get_current_energy()
        reward_before = self.get_current_reward()
        self.publish_policy_outcome(
            policy,
            started=started,
            finished=True,
            success=success,
            status=status,
            energy_before=energy_before,
            reward_before=reward_before,
        )

    def publish_policy_outcome(
        self,
        policy: RobotinoSelectedPolicy,
        started: bool,
        finished: bool,
        success: bool,
        status: str,
        energy_before: float,
        reward_before: float,
    ) -> None:
        outcome = RobotinoPolicyOutcome()
        outcome.header.stamp = self.get_clock().now().to_msg()
        outcome.header.frame_id = self.map_frame
        outcome.valid = True

        outcome.policy_id = int(policy.policy_id)
        outcome.policy_name = str(policy.policy_name)
        outcome.started = bool(started)
        outcome.finished = bool(finished)
        outcome.success = bool(success)
        outcome.status = str(status)
        outcome.target_tag_id = int(policy.target_tag_id)

        energy_after = self.get_current_energy()
        reward_after = self.get_current_reward()

        outcome.energy_before = float(energy_before)
        outcome.energy_after = float(energy_after)
        outcome.energy_delta = float(energy_after - energy_before)

        outcome.reward_before = float(reward_before)
        outcome.reward_after = float(reward_after)
        outcome.reward_delta = float(reward_after - reward_before)

        self.outcome_publisher.publish(outcome)

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