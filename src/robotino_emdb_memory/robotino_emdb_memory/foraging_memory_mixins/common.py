"""Shared numerical helpers and evidence calculations."""

import math

class MemoryCommonMixin:
    """Mixin extracted from RobotinoForagingMemory."""

    @staticmethod
    def clamp(value, min_value, max_value):
        return float(max(min_value, min(max_value, float(value))))

    @staticmethod
    def probability(positive, negative):
        try:
            positive = float(positive)
            negative = float(negative)
        except (TypeError, ValueError):
            return 0.5
        if not math.isfinite(positive) or not math.isfinite(negative):
            return 0.5
        positive = max(0.0, positive)
        negative = max(0.0, negative)
        total = positive + negative
        if total <= 0.0:
            return 0.5
        return positive / total

    @staticmethod
    def set_if_available(message, field_name, value):
        """Set optional fields after RobotinoForagingState.msg is extended."""
        if hasattr(message, field_name):
            setattr(message, field_name, value)

    def energy_need(self):
        if self.low_energy_threshold <= 0.0:
            return 0.0
        return self.clamp(
            (self.low_energy_threshold - self.robot_energy)
            / self.low_energy_threshold,
            0.0,
            1.0,
        )

    def default_evidence(self):
        return {
            # A tag enters memory because it was visually detected.
            "presence_positive": 2.0,
            "presence_negative": 1.0,
            # Reachability/recharge begin uncertain.
            "reachability_positive": 1.0,
            "reachability_negative": 1.0,
            "recharge_positive": 1.0,
            "recharge_negative": 1.0,
            "navigation_attempts": 0,
            "navigation_successes": 0,
            "navigation_failures": 0,
            "verification_attempts": 0,
            "verified_absences": 0,
            "recharge_attempts": 0,
            "recharge_successes": 0,
            "consecutive_navigation_failures": 0,
            "consecutive_not_found": 0,
            "consecutive_recharge_failures": 0,
            "last_recharge_attempt_time": 0.0,
            "status": "UNVERIFIED",
            "last_outcome": "none",
            "last_failure_reason": "none",
            "last_supply_reason": "unverified",
            "last_supply_required": 0.0,
            "last_supply_deliverable": 0.0,
        }

    def ensure_memory_schema(self, data):
        for key, value in self.default_evidence().items():
            data.setdefault(key, value)
        data.setdefault("position_samples", [])
        data["position_samples"] = [
            (float(sample[0]), float(sample[1]))
            for sample in data["position_samples"]
            if isinstance(sample, (list, tuple)) and len(sample) >= 2
        ]
        return data

    def presence_confidence(self, data):
        return self.probability(
            data["presence_positive"], data["presence_negative"]
        )

    def reachability_confidence(self, data):
        return self.probability(
            data["reachability_positive"],
            data["reachability_negative"],
        )

    def recharge_reliability(self, data):
        return self.probability(
            data["recharge_positive"], data["recharge_negative"]
        )

    def worthiness(self, data):
        return (
            self.presence_confidence(data)
            * self.reachability_confidence(data)
            * self.recharge_reliability(data)
        )

    def log_tag_reliability(self, tag_id, data):
        self.get_logger().info(
            "Tag %d | outcome=%s | status=%s | presence=%.3f | "
            "reachability=%.3f | recharge=%.3f | worthiness=%.3f"
            % (
                tag_id,
                data["last_outcome"],
                data["status"],
                self.presence_confidence(data),
                self.reachability_confidence(data),
                self.recharge_reliability(data),
                self.worthiness(data),
            )
        )
