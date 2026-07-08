import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():
    pkg_robotino_emdb_actuation = get_package_share_directory(
        'robotino_emdb_actuation'
    )

    nav2_params_file = os.path.join(
        pkg_robotino_emdb_actuation,
        'config',
        'nav2_params.yaml'
    )

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_bringup_dir,
                'launch',
                'bringup_launch.py'
            )
        ),
        launch_arguments={
            'params_file': nav2_params_file,
            'use_sim_time': 'true',
            'map': '/path/to/your/map.yaml',
        }.items()
    )

    return LaunchDescription([
        nav2_launch,
    ])