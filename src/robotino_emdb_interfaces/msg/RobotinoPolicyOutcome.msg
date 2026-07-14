# RobotinoPolicyOutcome.msg
#
# The policy executor publishes objective execution results.
# The LTM uses these results to update its memory.

std_msgs/Header header

# ------------------------------------------------------------------
# Policy result
# ------------------------------------------------------------------

uint8 POLICY_UNKNOWN=0
uint8 POLICY_EXPLORE=1
uint8 POLICY_RETURN_TO_ENERGY=2
uint8 POLICY_VERIFY_ENERGY=3
uint8 POLICY_GO_TO_GOAL=4

uint8 policy_id

bool policy_completed
bool policy_success

# ------------------------------------------------------------------
# Target
# ------------------------------------------------------------------

uint8 TARGET_NONE=0
uint8 TARGET_ENERGY_TAG=1
uint8 TARGET_GOAL_TAG=2
uint8 TARGET_FRONTIER=3

uint8 target_type

# AprilTag ID.
# Use -1 when there is no tag target.
int32 target_id

# ------------------------------------------------------------------
# Navigation result
# ------------------------------------------------------------------

uint8 NAV_NOT_USED=0
uint8 NAV_SUCCEEDED=1
uint8 NAV_FAILED=2
uint8 NAV_CANCELED=3

uint8 navigation_result

# ------------------------------------------------------------------
# Tag verification result
# ------------------------------------------------------------------

uint8 TAG_NOT_CHECKED=0
uint8 TAG_FOUND=1
uint8 TAG_NOT_FOUND=2
uint8 TAG_CHECK_INCONCLUSIVE=3
uint8 TAG_FOUND_AT_NEW_LOCATION=4

uint8 tag_result

# Confidence reported by the AprilTag detector when the target was found.
# Set to 0.0 when the tag was not detected.
float32 detection_confidence

# Used when the same tag is detected at a new location.
bool observed_tag_pose_valid
geometry_msgs/PoseStamped observed_tag_pose

# ------------------------------------------------------------------
# Energy result
# ------------------------------------------------------------------

bool recharge_attempted
bool recharge_succeeded

float32 energy_before
float32 energy_after

# ------------------------------------------------------------------
# Failure information
# ------------------------------------------------------------------

# Examples:
# "none"
# "path_unavailable"
# "navigation_aborted"
# "tag_not_detected"
# "observation_inconclusive"
# "energy_not_received"
string failure_reason