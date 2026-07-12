import math

def yaw_to_quaternion_z_w(yaw):
    return math.sin(float(yaw) / 2.0), math.cos(float(yaw) / 2.0)

def euclidean_distance(x1, y1, x2, y2):
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    return math.sqrt(dx * dx + dy * dy)


import math


def compute_tag_approach_pose(
    robot_x,
    robot_y,
    tag_x,
    tag_y,
    normal_x,
    normal_y,
    standoff=0.65,
):
    """
    Returns:
        goal_x, goal_y, goal_yaw

    standoff is the distance between the tag/wall and base_link.
    """

    normal_length = math.hypot(normal_x, normal_y)

    if normal_length < 1e-6:
        raise ValueError("Tag normal vector has zero length")

    normal_x /= normal_length
    normal_y /= normal_length

    # Two possible sides of the tag.
    candidate_1 = (
        tag_x + standoff * normal_x,
        tag_y + standoff * normal_y,
    )

    candidate_2 = (
        tag_x - standoff * normal_x,
        tag_y - standoff * normal_y,
    )

    # Initially select the candidate on the robot's side of the wall.
    distance_1 = math.hypot(
        candidate_1[0] - robot_x,
        candidate_1[1] - robot_y,
    )

    distance_2 = math.hypot(
        candidate_2[0] - robot_x,
        candidate_2[1] - robot_y,
    )

    if distance_1 <= distance_2:
        goal_x, goal_y = candidate_1
    else:
        goal_x, goal_y = candidate_2

    # Make the robot face the tag when it reaches the goal.
    goal_yaw = math.atan2(
        tag_y - goal_y,
        tag_x - goal_x,
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

def yaw_to_quaternion_z_w(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)