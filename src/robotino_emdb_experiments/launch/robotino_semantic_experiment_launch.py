#!/usr/bin/env python3

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context: LaunchContext, *args, **kwargs):
    """Own every Robotino/e-MDB integration node exactly once."""

    detections_topic = LaunchConfiguration("detections_topic")
    output_topic = LaunchConfiguration("output_topic")
    map_frame = LaunchConfiguration("map_frame")
    robot_frame = LaunchConfiguration("robot_frame")
    camera_frame = LaunchConfiguration("camera_frame")
    semantics_file = LaunchConfiguration("semantics_file")
    mapping_complete_topic = LaunchConfiguration('mapping_complete_topic')
    mapping_progress_topic = LaunchConfiguration('mapping_progress_topic') 

    enable_nav2_execution = LaunchConfiguration("enable_nav2_execution")
    nav2_action_name = LaunchConfiguration("nav2_action_name")
    publish_exploration_control = LaunchConfiguration(
        "publish_exploration_control"
    )

    log_level = LaunchConfiguration("log_level")
    random_seed = LaunchConfiguration("random_seed")
    config_package = LaunchConfiguration("config_package")
    config_file = LaunchConfiguration("config_file")
    experiment_package = LaunchConfiguration("experiment_package")
    experiment_file = LaunchConfiguration("experiment_file")

    # Official GII e-MDB runtime.
    commander_node = Node(
        package="core",
        executable="commander",
        name="commander",
        output="screen",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[{"random_seed": random_seed}],
    )

    ltm_node = Node(
        package="core",
        executable="ltm",
        name="ltm",
        output="screen",
        arguments=["0", "--ros-args", "--log-level", log_level],
    )

    load_commander_config = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "/commander/load_config ",
                "core_interfaces/srv/LoadConfig ",
                '"{file: ',
                PathJoinSubstitution(
                    [
                        FindPackageShare(config_package),
                        "config",
                        config_file,
                    ]
                ),
                '}"',
            ]
        ],
        shell=True,
        output="screen",
    )

    load_experiment_config = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "/commander/load_experiment ",
                "core_interfaces/srv/LoadConfig ",
                '"{file: ',
                PathJoinSubstitution(
                    [
                        FindPackageShare(experiment_package),
                        "experiments",
                        experiment_file,
                    ]
                ),
                '}"',
            ]
        ],
        shell=True,
        output="screen",
    )

    # Robotino sensor-to-memory chain.
    apriltag_tf_bridge_node = Node(
        package="robotino_emdb_bridge",
        executable="apriltag_tf_to_emdb_bridge",
        name="apriltag_tf_to_emdb_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "detections_topic": detections_topic,
                "output_topic": output_topic,
                "map_frame": map_frame,
                "robot_frame": robot_frame,
                "camera_frame": camera_frame,
            }
        ],
    )

    # 4. Factual-state adapter for official GII DriveExponential nodes.
    cognitive_signals_node = Node(
        package='robotino_emdb_motivation',
        executable='cognitive_signals',
        name='robotino_cognitive_signals',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'foraging_topic': '/robotino/emdb/foraging_state',
            'mapping_complete_topic': mapping_complete_topic,
            'mapping_progress_topic': mapping_progress_topic,
            'energy_satisfaction_topic':
                '/robotino/emdb/satisfaction/energy',
            'resource_satisfaction_topic':
                '/robotino/emdb/satisfaction/resource_knowledge',
            'exploration_satisfaction_topic':
                '/robotino/emdb/satisfaction/exploration',
            'mission_satisfaction_topic':
                '/robotino/emdb/satisfaction/mission',
            'publish_period_s': 0.2,
        }]
    )

    foraging_memory_node = Node(
        package="robotino_emdb_memory",
        executable="foraging_memory",
        name="foraging_memory",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "input_topic": output_topic,
                "outcome_topic": "/robotino/emdb/policy_outcome",
                "output_topic": "/robotino/emdb/foraging_state",
                "semantics_file": semantics_file,
            }
        ],
    )

    # Official e-MDB policy service to the existing Robotino executor.
    policy_execution_bridge_node = Node(
        package="robotino_emdb_decision",
        executable="policy_execution_bridge",
        name="robotino_policy_execution_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "service_name": "/robotino/emdb/execute_policy",
                "foraging_topic": "/robotino/emdb/foraging_state",
                "selected_policy_topic": "/robotino/emdb/selected_policy",
                "outcome_topic": "/robotino/emdb/policy_outcome",
                "execution_timeout_s": 120.0,
                "minimum_energy_bank_score": 0.0,
            }
        ],
    )

    policy_executor_node = Node(
        package="robotino_emdb_actuation",
        executable="policy_executor",
        name="robotino_policy_executor",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "policy_topic": "/robotino/emdb/selected_policy",
                "foraging_topic": "/robotino/emdb/foraging_state",
                "outcome_topic": "/robotino/emdb/policy_outcome",
                "nav2_action_name": nav2_action_name,
                "map_frame": map_frame,
                "enable_nav2_execution": ParameterValue(
                    enable_nav2_execution,
                    value_type=bool,
                ),
                "publish_exploration_control": ParameterValue(
                    publish_exploration_control,
                    value_type=bool,
                ),
            }
        ],
    )

    shutdown_on_commander_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=commander_node,
            on_exit=[Shutdown()],
        )
    )

    return [
        commander_node,
        ltm_node,
        shutdown_on_commander_exit,
        TimerAction(period=1.0, actions=[apriltag_tf_bridge_node]),
        TimerAction(period=2.0, actions=[foraging_memory_node]),
        TimerAction(period=2.0, actions=[load_commander_config]),
        TimerAction(period=3.0,actions=[cognitive_signals_node]), 
        TimerAction(period=3.0, actions=[policy_execution_bridge_node]),
        TimerAction(period=6.0, actions=[policy_executor_node]),
        TimerAction(period=5.0, actions=[load_experiment_config]),
    ]

def generate_launch_description():
    default_semantics_file = PathJoinSubstitution(
        [
            FindPackageShare("robotino_emdb_memory"),
            "config",
            "foraging_semantics.yaml",
        ]
    )

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
            DeclareLaunchArgument(
                "mapping_complete_topic",
                default_value="/frontier_exploration/mapping_complete",
                description=(
                    "std_msgs/Bool topic published when frontier mapping is complete."
                ),
            ),

            DeclareLaunchArgument(
                "mapping_progress_topic",
                default_value="",
                description=(
                    "Optional std_msgs/Float32 mapping progress topic in [0, 1]. "
                    "Leave empty when only mapping_complete is available."
                ),
            ),
                        OpaqueFunction(function=launch_setup),
        ]
    )