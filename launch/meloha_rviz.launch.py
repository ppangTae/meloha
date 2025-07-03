import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_vive_tracker = LaunchConfiguration('use_vive_tracker', default='true')

    declare_use_sim  = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock')

    declare_use_vive = DeclareLaunchArgument(
        'use_vive_tracker', default_value='false',
        description='If true, assume /joint_states comes from Vive Tracker '
                    'and skip joint_state_publisher_gui')

    urdf_path = PathJoinSubstitution([
        get_package_share_directory('meloha'),
        'urdf', 'simple_meloha.urdf.xml'
    ])
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    rviz_config_path = PathJoinSubstitution([
        get_package_share_directory('meloha'),
        'rviz',
        'meloha.rviz'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf_path]),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time
            }]
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_path],
            output='screen'
        )
    ])