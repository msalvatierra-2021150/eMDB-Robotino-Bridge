from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context: LaunchContext, *args, **kwargs):
    logger = LaunchConfiguration("log_level")
    random_seed = LaunchConfiguration("random_seed")

    experiment_file = LaunchConfiguration("experiment_file")
    experiment_package = LaunchConfiguration("experiment_package")

    config_file = LaunchConfiguration("config_file")
    config_package = LaunchConfiguration("config_package")

    commander_node = Node(
        package="core",
        executable="commander",
        output="screen",
        arguments=["--ros-args", "--log-level", logger],
        parameters=[{"random_seed": random_seed}],
    )

    ltm_node = Node(
        package="core",
        executable="ltm",
        output="screen",
        arguments=["0", "--ros-args", "--log-level", logger],
    )

    # 1) Load commander.yaml first.
    # This creates the execution nodes.
    load_commander_config = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "commander/load_config ",
                "core_interfaces/srv/LoadConfig ",
                '"{file: ',
                PathJoinSubstitution([
                    FindPackageShare(config_package),
                    "config",
                    config_file,
                ]),
                '}"',
            ]
        ],
        shell=True,
        output="screen",
    )

    # 2) Load your experiment YAML second.
    # This creates your Perception node, Drive nodes, MainLoop, etc.
    load_experiment_config = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                "commander/load_experiment ",
                "core_interfaces/srv/LoadConfig ",
                '"{file: ',
                PathJoinSubstitution([
                    FindPackageShare(experiment_package),
                    "experiments",
                    experiment_file,
                ]),
                '}"',
            ]
        ],
        shell=True,
        output="screen",
    )

    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=commander_node,
            on_exit=[Shutdown()],
        )
    )

    return [
        commander_node,
        ltm_node,

        TimerAction(
            period=2.0,
            actions=[load_commander_config],
        ),

        TimerAction(
            period=5.0,
            actions=[load_experiment_config],
        ),

        shutdown_on_exit,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="Logging level",
        ),

        DeclareLaunchArgument(
            "random_seed",
            default_value="0",
            description="Random seed",
        ),

        DeclareLaunchArgument(
            "config_package",
            default_value="core",
            description="Package containing commander.yaml",
        ),

        DeclareLaunchArgument(
            "config_file",
            default_value="commander.yaml",
            description="Commander config YAML",
        ),

        DeclareLaunchArgument(
            "experiment_package",
            default_value="robotino_emdb_experiments",
            description="Package containing the experiment YAML",
        ),

        DeclareLaunchArgument(
            "experiment_file",
            default_value="robotino_semantic_experiment.yaml",
            description="Robotino experiment YAML",
        ),

        OpaqueFunction(function=launch_setup),
    ])