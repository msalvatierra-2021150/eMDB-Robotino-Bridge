#!/usr/bin/env python3
"""Robotino-specific semantic/resource memory for the GII e-MDB integration.

The node stores factual tag/resource memory while the official e-MDB LTM owns
cognitive nodes and learned policy relations.  Implementation details are split
into mixins so ranking, resource simulation, persistence, observations, and
policy-outcome evidence can evolve independently.
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoPolicyOutcome,
    RobotinoTag,
)

try:
    from .foraging_memory_mixins import (
        EnergyBankRankingMixin,
        MemoryCommonMixin,
        PolicyOutcomeMixin,
        ResourceSimulationMixin,
        SemanticsPersistenceMixin,
        StateObservationMixin,
    )
except ImportError:  # Direct execution during local debugging.
    from foraging_memory_mixins import (
        EnergyBankRankingMixin,
        MemoryCommonMixin,
        PolicyOutcomeMixin,
        ResourceSimulationMixin,
        SemanticsPersistenceMixin,
        StateObservationMixin,
    )


class RobotinoForagingMemory(
    StateObservationMixin,
    PolicyOutcomeMixin,
    EnergyBankRankingMixin,
    ResourceSimulationMixin,
    SemanticsPersistenceMixin,
    MemoryCommonMixin,
    Node,
):
    """Maintain Robotino's factual resource registry and reliability evidence."""

    def __init__(self):
        super().__init__("robotino_foraging_memory")

        # ------------------------------------------------------------------
        # Files and topics
        # ------------------------------------------------------------------
        self.declare_parameter(
            "semantics_file",
            "/home/mike/eMDB_ws/src/robotino_emdb_memory/"
            "config/foraging_semantics.yaml",
        )
        self.declare_parameter(
            "memory_file",
            "~/.robotino_emdb/robotino_resource_memory.yaml",
        )
        self.declare_parameter("persist_memory", True)
        self.declare_parameter(
            "input_topic", "/robotino/emdb/tag_observation"
        )
        self.declare_parameter(
            "outcome_topic", "/robotino/emdb/policy_outcome"
        )
        self.declare_parameter(
            "output_topic", "/robotino/emdb/foraging_state"
        )
        self.declare_parameter(
            "recharge_target_topic",
            "/robotino/emdb/recharge_target",
        )

        # ------------------------------------------------------------------
        # Robot energy and recovery-aware bank eligibility
        # ------------------------------------------------------------------
        self.declare_parameter("initial_energy", 1.0)
        self.declare_parameter("energy_decay_per_second", 0.005)
        self.declare_parameter("low_energy_threshold", 0.35)
        self.declare_parameter("resume_energy_threshold", 0.50)

        # By default, any bank with usable energy may contribute to recovery.
        # Recovery may span multiple banks; one bank does not need to close the
        # full gap to resume_energy_threshold by itself.
        self.declare_parameter("require_full_recovery_supply", False)
        self.declare_parameter("resource_empty_epsilon", 0.01)
        self.declare_parameter("actionable_supply_epsilon", 0.005)
        self.declare_parameter("recovery_supply_margin", 0.0)

        # During energy recovery, only banks that can complete recovery may be
        # published as best_energy_tag_id. This forces search_for_energy when
        # every remembered bank is empty or insufficient. Outside recovery,
        # delayed verification candidates can still support renewable-resource
        # learning without trapping the Robotino at a known depleted bank.
        self.declare_parameter(
            "allow_nonactionable_verification_during_recovery", False
        )
        self.declare_parameter("depleted_verification_score_factor", 0.10)
        self.declare_parameter("depleted_retry_delay_s", 60.0)
        self.declare_parameter("insufficient_verification_score_factor", 0.05)
        self.declare_parameter("insufficient_supply_retry_delay_s", 60.0)

        self.declare_parameter("arrival_distance", 1.2)
        # A drained renewable bank is unavailable for the current visit until
        # Robotino physically leaves its vicinity. Hidden regeneration may
        # continue, but regenerated energy belongs to a future visit.
        self.declare_parameter("recharge_rearm_distance", 1.55)
        self.declare_parameter("same_tag_event_gap", 3.0)
        self.declare_parameter("state_publish_rate_hz", 5.0)

        # Evidence weights for the learned factual outcomes.
        self.declare_parameter("successful_recharge_weight", 2.0)
        self.declare_parameter("failed_recharge_weight", 1.0)
        self.declare_parameter("navigation_failure_weight", 1.0)
        self.declare_parameter("verified_absence_weight", 1.0)
        self.declare_parameter("missing_confidence_threshold", 0.35)
        self.declare_parameter("unreachable_confidence_threshold", 0.35)
        self.declare_parameter("missing_after_consecutive_failures", 2)
        self.declare_parameter("unreachable_after_consecutive_failures", 2)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.outcome_topic = str(self.get_parameter("outcome_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.recharge_target_topic = str(
            self.get_parameter("recharge_target_topic").value
        )

        self.robot_energy = self.clamp(
            self.get_parameter("initial_energy").value,
            0.0,
            1.0,
        )
        self.energy_decay_per_second = max(
            0.0,
            float(self.get_parameter("energy_decay_per_second").value),
        )
        self.low_energy_threshold = self.clamp(
            self.get_parameter("low_energy_threshold").value,
            0.0,
            1.0,
        )
        self.resume_energy_threshold = self.clamp(
            self.get_parameter("resume_energy_threshold").value,
            self.low_energy_threshold,
            1.0,
        )
        self.require_full_recovery_supply = bool(
            self.get_parameter("require_full_recovery_supply").value
        )
        self.resource_empty_epsilon = max(
            0.0,
            float(self.get_parameter("resource_empty_epsilon").value),
        )
        self.actionable_supply_epsilon = max(
            0.0,
            float(self.get_parameter("actionable_supply_epsilon").value),
        )
        self.recovery_supply_margin = max(
            0.0,
            float(self.get_parameter("recovery_supply_margin").value),
        )

        self.allow_nonactionable_verification_during_recovery = bool(
            self.get_parameter(
                "allow_nonactionable_verification_during_recovery"
            ).value
        )
        self.depleted_verification_score_factor = self.clamp(
            self.get_parameter("depleted_verification_score_factor").value,
            0.0,
            1.0,
        )
        self.depleted_retry_delay_s = max(
            0.0,
            float(self.get_parameter("depleted_retry_delay_s").value),
        )
        self.insufficient_verification_score_factor = self.clamp(
            self.get_parameter("insufficient_verification_score_factor").value,
            0.0,
            1.0,
        )
        self.insufficient_supply_retry_delay_s = max(
            0.0,
            float(self.get_parameter("insufficient_supply_retry_delay_s").value),
        )

        self.arrival_distance = max(
            0.0,
            float(self.get_parameter("arrival_distance").value),
        )
        self.recharge_rearm_distance = max(
            self.arrival_distance + 0.05,
            float(self.get_parameter("recharge_rearm_distance").value),
        )
        self.same_tag_event_gap = max(
            0.0,
            float(self.get_parameter("same_tag_event_gap").value),
        )
        self.state_publish_rate_hz = max(
            0.2,
            float(self.get_parameter("state_publish_rate_hz").value),
        )

        self.successful_recharge_weight = float(
            self.get_parameter("successful_recharge_weight").value
        )
        self.failed_recharge_weight = float(
            self.get_parameter("failed_recharge_weight").value
        )
        self.navigation_failure_weight = float(
            self.get_parameter("navigation_failure_weight").value
        )
        self.verified_absence_weight = float(
            self.get_parameter("verified_absence_weight").value
        )
        self.missing_confidence_threshold = float(
            self.get_parameter("missing_confidence_threshold").value
        )
        self.unreachable_confidence_threshold = float(
            self.get_parameter("unreachable_confidence_threshold").value
        )
        self.missing_after_consecutive_failures = int(
            self.get_parameter("missing_after_consecutive_failures").value
        )
        self.unreachable_after_consecutive_failures = int(
            self.get_parameter("unreachable_after_consecutive_failures").value
        )

        self.persist_memory = bool(
            self.get_parameter("persist_memory").value
        )
        self.memory_file = Path(
            str(self.get_parameter("memory_file").value)
        ).expanduser()

        # ------------------------------------------------------------------
        # Semantic and factual memory
        # ------------------------------------------------------------------
        self.semantics_file = str(
            self.get_parameter("semantics_file").value
        )
        self.tag_semantics = self.load_tag_semantics(self.semantics_file)
        self.warned_unknown_tag_ids = set()
        self.resource_truth = self.create_resource_truth()

        energy_tag_ids = sorted(
            tag_id
            for tag_id, semantics in self.tag_semantics.items()
            if bool(semantics.get("is_energy_bank", False))
        )
        self.get_logger().info(
            f"Configured energy-bank tag IDs: {energy_tag_ids}"
        )
        if not energy_tag_ids:
            self.get_logger().error(
                "No energy-bank tags are configured. "
                "Check foraging_semantics.yaml."
            )

        self.memory = {}
        self.load_resource_memory()
        self.restore_resource_truth_from_memory()

        self.goal_satisfied = False
        self.active_recharge_target_id = -1
        # Prevent repeated publications of the same authorization command from
        # restarting collection as a just-drained bank regenerates by crumbs.
        # The executor's explicit -1 command clears this interaction latch.
        self.blocked_recharge_target_id = -1
        self.last_logged_best_energy_tag_id = None
        self.last_best_candidate_summary = None
        self.latest_state = self.create_empty_state()

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.observation_subscriber = self.create_subscription(
            RobotinoTag,
            self.input_topic,
            self.observation_callback,
            10,
        )
        self.outcome_subscriber = self.create_subscription(
            RobotinoPolicyOutcome,
            self.outcome_topic,
            self.outcome_callback,
            10,
        )
        self.recharge_target_subscriber = self.create_subscription(
            Int32,
            self.recharge_target_topic,
            self.recharge_target_callback,
            10,
        )
        self.publisher = self.create_publisher(
            RobotinoForagingState,
            self.output_topic,
            10,
        )

        self.energy_timer = self.create_timer(1.0, self.energy_decay_step)
        self.state_timer = self.create_timer(
            1.0 / self.state_publish_rate_hz,
            self.publish_current_state,
        )
        self.persistence_timer = self.create_timer(
            5.0,
            self.save_resource_memory,
        )

        self.get_logger().info("Robotino resource memory started")
        self.get_logger().info(f"Tag observations: {self.input_topic}")
        self.get_logger().info(f"Policy outcomes: {self.outcome_topic}")
        self.get_logger().info(
            f"Recharge authorization: {self.recharge_target_topic}"
        )
        self.get_logger().info(f"Foraging state: {self.output_topic}")
        self.get_logger().info(f"Persistent memory: {self.memory_file}")
        self.get_logger().info(
            "Recovery-aware bank selection: "
            f"resume={self.resume_energy_threshold:.3f}, "
            f"require_full_supply={self.require_full_recovery_supply}, "
            f"empty_epsilon={self.resource_empty_epsilon:.3f}, "
            f"rearm_distance={self.recharge_rearm_distance:.2f}m, "
            f"insufficient_retry={self.insufficient_supply_retry_delay_s:.1f}s"
        )

    def destroy_node(self):
        self.save_resource_memory()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotinoForagingMemory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()