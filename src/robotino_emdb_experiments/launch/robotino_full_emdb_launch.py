#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    detections_topic = LaunchConfiguration('detections_topic')
    output_topic = LaunchConfiguration('output_topic')
    map_frame = LaunchConfiguration('map_frame')
    robot_frame = LaunchConfiguration('robot_frame')
    camera_frame = LaunchConfiguration('camera_frame')
    semantics_file = LaunchConfiguration('semantics_file')

    enable_nav2_execution = LaunchConfiguration('enable_nav2_execution')
    nav2_action_name = LaunchConfiguration('nav2_action_name')
    publish_exploration_control = LaunchConfiguration('publish_exploration_control')

    semantic_experiment_launch = PathJoinSubstitution([
        FindPackageShare('robotino_emdb_experiments'),
        'launch',
        'robotino_semantic_experiment_launch.py'
    ])

    default_semantics_file = PathJoinSubstitution([
        FindPackageShare('robotino_emdb_memory'),
        'config',
        'foraging_semantics.yaml'
    ])

    # 1. Bridge: /detections + /tf -> /robotino/emdb/tag_observation
    apriltag_tf_bridge_node = Node(
        package='robotino_emdb_bridge',
        executable='apriltag_tf_to_emdb_bridge',
        name='apriltag_tf_to_emdb_bridge',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'detections_topic': detections_topic,
            'output_topic': output_topic,
            'map_frame': map_frame,
            'robot_frame': robot_frame,
            'camera_frame': camera_frame,
        }]
    )

    # 2. e-MDB experiment: commander, LTM, perception nodes, main loop
    semantic_experiment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(semantic_experiment_launch)
    )

    # 3. Episodic/resource memory:
    # /robotino/emdb/tag_observation -> /robotino/emdb/foraging_state
    foraging_memory_node = Node(
        package='robotino_emdb_memory',
        executable='foraging_memory',
        name='foraging_memory',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_topic': '/robotino/emdb/tag_observation',
            'output_topic': '/robotino/emdb/foraging_state',
            'semantics_file': semantics_file,
        }]
    )

    # 4. Motivational system:
    # /robotino/emdb/foraging_state -> /robotino/emdb/motivation_state
    motivational_system_node = Node(
        package='robotino_emdb_motivation',
        executable='motivational_system',
        name='robotino_motivational_system',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_topic': '/robotino/emdb/foraging_state',
            'output_topic': '/robotino/emdb/motivation_state',
        }]
    )

    # 5. Decision / policy selector:
    # /robotino/emdb/motivation_state -> /robotino/emdb/selected_policy
    policy_selector_node = Node(
        package='robotino_emdb_decision',
        executable='policy_selector',
        name='robotino_policy_selector',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_topic': '/robotino/emdb/motivation_state',
            'output_topic': '/robotino/emdb/selected_policy',
        }]
    )

    # 6. Actuation / policy executor:
    # /robotino/emdb/selected_policy -> Nav2 / exploration / cmd_vel
    policy_executor_node = Node(
        package='robotino_emdb_actuation',
        executable='policy_executor',
        name='robotino_policy_executor',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'policy_topic': '/robotino/emdb/selected_policy',
            'foraging_topic': '/robotino/emdb/foraging_state',
            'outcome_topic': '/robotino/emdb/policy_outcome',

            'nav2_action_name': nav2_action_name,
            'map_frame': map_frame,

            # IMPORTANT:
            # Use ParameterValue so launch converts "true"/"false" into real booleans.
            'enable_nav2_execution': ParameterValue(
                enable_nav2_execution,
                value_type=bool
            ),
            'publish_exploration_control': ParameterValue(
                publish_exploration_control,
                value_type=bool
            ),
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/detections',
            description='AprilTag detections topic.'
        ),

        DeclareLaunchArgument(
            'output_topic',
            default_value='/robotino/emdb/tag_observation',
            description='Output topic for semantic tag observations.'
        ),

        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Global map frame.'
        ),

        DeclareLaunchArgument(
            'robot_frame',
            default_value='base_link',
            description='Robot base frame.'
        ),

        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_optical_frame',
            description='Camera optical frame.'
        ),

        DeclareLaunchArgument(
            'semantics_file',
            default_value=default_semantics_file,
            description='YAML file defining semantic tag meanings.'
        ),

        DeclareLaunchArgument(
            'enable_nav2_execution',
            default_value='true',
            description='If true, policy executor sends Nav2 goals.'
        ),

        DeclareLaunchArgument(
            'nav2_action_name',
            default_value='navigate_to_pose',
            description='Nav2 NavigateToPose action name.'
        ),

        DeclareLaunchArgument(
            'publish_exploration_control',
            default_value='false',
            description='If true, publishes exploration enable/disable commands.'
        ),

        # 1. Start AprilTag TF to e-MDB bridge
        apriltag_tf_bridge_node,

        # 2. Start foraging semantic memory
        TimerAction(
            period=1.0,
            actions=[
                foraging_memory_node
            ]
        ),

        # 3. Start motivational system
        TimerAction(
            period=2.0,
            actions=[
                motivational_system_node
            ]
        ),

        # 4. Start policy selector
        TimerAction(
            period=3.0,
            actions=[
                policy_selector_node
            ]
        ),

        # 5. Start policy executor / actuation
        TimerAction(
            period=4.0,
            actions=[
                policy_executor_node
            ]
        ),

        # 6. Start the semantic/e-MDB experiment
        TimerAction(
            period=5.0,
            actions=[
                semantic_experiment
            ]
        ),
    ])