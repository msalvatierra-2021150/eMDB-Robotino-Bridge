#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Top-level wrapper.

    This file starts no nodes directly. It includes
    robotino_semantic_experiment_launch.py exactly once and forwards all
    configuration arguments to it.
    """

    semantic_experiment_launch = PathJoinSubstitution(
        [
            FindPackageShare("robotino_emdb_experiments"),
            "launch",
            "robotino_semantic_experiment_launch.py",
        ]
    )

    default_semantics_file = PathJoinSubstitution(
        [
            FindPackageShare("robotino_emdb_memory"),
            "config",
            "foraging_semantics.yaml",
        ]
    )

    forwarded_arguments = {
        "detections_topic": LaunchConfiguration("detections_topic"),
        "output_topic": LaunchConfiguration("output_topic"),
        "map_frame": LaunchConfiguration("map_frame"),
        "robot_frame": LaunchConfiguration("robot_frame"),
        "camera_frame": LaunchConfiguration("camera_frame"),
        "semantics_file": LaunchConfiguration("semantics_file"),
        "mapping_complete_topic": LaunchConfiguration("mapping_complete_topic"),
        "mapping_progress_topic": LaunchConfiguration("mapping_progress_topic"),
        "enable_nav2_execution": LaunchConfiguration("enable_nav2_execution"),
        "nav2_action_name": LaunchConfiguration("nav2_action_name"),
        "publish_exploration_control": LaunchConfiguration(
            "publish_exploration_control"
        ),
        "log_level": LaunchConfiguration("log_level"),
        "random_seed": LaunchConfiguration("random_seed"),
        "config_package": LaunchConfiguration("config_package"),
        "config_file": LaunchConfiguration("config_file"),
        "experiment_package": LaunchConfiguration("experiment_package"),
        "experiment_file": LaunchConfiguration("experiment_file"),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/detections",
                description="AprilTag detections topic.",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/robotino/emdb/tag_observation",
                description="Semantic tag-observation output topic.",
            ),
            DeclareLaunchArgument(
                "map_frame",
                default_value="map",
                description="Global map frame.",
            ),
            DeclareLaunchArgument(
                "robot_frame",
                default_value="base_link",
                description="Robot base frame.",
            ),
            DeclareLaunchArgument(
                "camera_frame",
                default_value="camera_optical_frame",
                description="Camera optical frame.",
            ),
            DeclareLaunchArgument(
                "semantics_file",
                default_value=default_semantics_file,
                description="YAML file defining semantic tag meanings.",
            ),
            DeclareLaunchArgument(
                "mapping_complete_topic",
                default_value="/frontier_exploration/mapping_complete",
                description="Bool topic indicating frontier exploration is complete.",
            ),
            DeclareLaunchArgument(
                "mapping_progress_topic",
                default_value="",
                description="Optional Float32 topic with exploration progress in [0, 1].",
            ),
            DeclareLaunchArgument(
                "enable_nav2_execution",
                default_value="true",
                description="Whether the policy executor sends Nav2 goals.",
            ),
            DeclareLaunchArgument(
                "nav2_action_name",
                default_value="navigate_to_pose",
                description="Nav2 NavigateToPose action name.",
            ),
            DeclareLaunchArgument(
                "publish_exploration_control",
                default_value="false",
                description=(
                    "Whether the executor publishes exploration enable/disable "
                    "commands."
                ),
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="GII e-MDB logging level.",
            ),
            DeclareLaunchArgument(
                "random_seed",
                default_value="0",
                description="Random seed used by the GII commander.",
            ),
            DeclareLaunchArgument(
                "config_package",
                default_value="core",
                description="Package containing the GII commander config.",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="commander.yaml",
                description="GII commander configuration file.",
            ),
            DeclareLaunchArgument(
                "experiment_package",
                default_value="robotino_emdb_experiments",
                description="Package containing the Robotino experiment YAML.",
            ),
            DeclareLaunchArgument(
                "experiment_file",
                default_value="robotino_semantic_experiment.yaml",
                description="Robotino e-MDB experiment YAML.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(semantic_experiment_launch),
                launch_arguments=forwarded_arguments.items(),
            ),
        ]
    )