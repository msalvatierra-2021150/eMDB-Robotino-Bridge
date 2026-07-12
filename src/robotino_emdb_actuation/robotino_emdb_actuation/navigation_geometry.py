import math

def yaw_to_quaternion_z_w(yaw):
    return math.sin(float(yaw) / 2.0), math.cos(float(yaw) / 2.0)

def euclidean_distance(x1, y1, x2, y2):
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    return math.sqrt(dx * dx + dy * dy)


def compute_approach_pose(robot_x, robot_y, target_x, target_y, standoff, epsilon):
    """
    Compute a Nav2 approach pose for a remembered target.

    The target is the actual tag/object position in map frame.
    The output is the pose where the robot should stand.
    """

    dx = float(target_x) - float(robot_x)
    dy = float(target_y) - float(robot_y)
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < epsilon:
        return float(robot_x), float(robot_y), 0.0

    ux = dx / dist
    uy = dy / dist

    goal_x = float(target_x) - float(standoff) * ux
    goal_y = float(target_y) - float(standoff) * uy

    goal_yaw = math.atan2(
        float(target_y) - goal_y,
        float(target_x) - goal_x,
    )

    return goal_x, goal_y, goal_yaw

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(quaternion):
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )

    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )

    return math.atan2(sin_yaw, cos_yaw)