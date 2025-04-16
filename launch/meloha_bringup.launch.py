
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)

from launch.conditions import (
  IfCondition,
)

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def launch_setup(context, *args, **kwargs):

    # # Dynamixel node setting
    # dynamixel_node = Node(
    #     package="dynamixel_sdk_examples",
    #     executable="read_write_node",
    #     namespace="dynamixel",
    #     name="dynamixel"
    # )
    
    rs_triple_camera_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),  # rs_triple_camera_launch.py가 있는 패키지명
                'launch',
                'rs_triple_camera_launch.py'
            ])
        ]),
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up MELOHA with the following launch configurations: ',

    ])

    return [
        rs_triple_camera_include,
        loginfo_action,
    ]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'launch_vive_tracker',
            default_value='true',
            choices=('true', 'false'),
            description=(
                'if `true`, launches both the leader and follower arms; '
                'if `false`, just the followers are launched'
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_high_name',
            default_value='cam_high',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_left_wrist_name',
            default_value='cam_left_wrist',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'cam_right_wrist_name',
            default_value='cam_right_wrist',
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
