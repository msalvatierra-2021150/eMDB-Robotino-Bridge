"""String constants used by PolicyOutcome.failure_reason.

The numeric policy, target, navigation, and tag-result values are generated
from PolicyOutcome.msg. Use PolicyOutcome.POLICY_*, TARGET_*, NAV_*, and
TAG_* directly instead of duplicating those integer values here.
"""

from typing import Final


# No failure occurred. This is also used while a policy is still in progress.
FAILURE_NONE: Final[str] = "none"
FAILURE_UNKNOWN: Final[str] = "unknown"

# Policy/executor failures.
FAILURE_POLICY_NOT_SUPPORTED: Final[str] = "policy_not_supported"
FAILURE_POLICY_DOES_NOT_USE_NAV2: Final[str] = "policy_does_not_use_nav2"
FAILURE_POLICY_PREEMPTED: Final[str] = "policy_preempted"
FAILURE_EXECUTION_DISABLED: Final[str] = "execution_disabled"
FAILURE_EXECUTOR_BUSY: Final[str] = "executor_busy"

# Target and planning failures.
FAILURE_NO_TARGET: Final[str] = "no_target"
FAILURE_INVALID_TARGET: Final[str] = "invalid_target"
FAILURE_TARGET_POSE_INVALID: Final[str] = "target_pose_invalid"
FAILURE_TARGET_UNREACHABLE: Final[str] = "target_unreachable"
FAILURE_NO_ROBOT_POSE: Final[str] = "no_robot_pose"
FAILURE_PATH_UNAVAILABLE: Final[str] = "path_unavailable"

# Navigation failures.
FAILURE_GOAL_REJECTED: Final[str] = "goal_rejected"
FAILURE_NAVIGATION_ABORTED: Final[str] = "navigation_aborted"
FAILURE_NAVIGATION_CANCELED: Final[str] = "navigation_canceled"
FAILURE_NAVIGATION_TIMEOUT: Final[str] = "navigation_timeout"
FAILURE_NAVIGATION_FAILED: Final[str] = "navigation_failed"

# Tag/interaction failures.
FAILURE_TAG_NOT_DETECTED: Final[str] = "tag_not_detected"
FAILURE_WRONG_TAG_DETECTED: Final[str] = "wrong_tag_detected"
FAILURE_OBSERVATION_INCONCLUSIVE: Final[str] = "observation_inconclusive"
FAILURE_OBSERVATION_TIMEOUT: Final[str] = "observation_timeout"
FAILURE_CAMERA_DATA_UNAVAILABLE: Final[str] = "camera_data_unavailable"
FAILURE_TAG_POSE_UNAVAILABLE: Final[str] = "tag_pose_unavailable"
FAILURE_GOAL_NOT_CONFIRMED: Final[str] = "goal_not_confirmed"

# Recharge failures.
FAILURE_RECHARGE_NOT_ATTEMPTED: Final[str] = "recharge_not_attempted"
FAILURE_ENERGY_NOT_RECEIVED: Final[str] = "energy_not_received"
FAILURE_RECHARGE_TIMEOUT: Final[str] = "recharge_timeout"
FAILURE_RECHARGE_FAILED: Final[str] = "recharge_failed"
FAILURE_ENERGY_DATA_UNAVAILABLE: Final[str] = "energy_data_unavailable"


ALL_FAILURE_REASONS: Final[frozenset[str]] = frozenset(
    value
    for name, value in globals().items()
    if name.startswith("FAILURE_") and isinstance(value, str)
)
