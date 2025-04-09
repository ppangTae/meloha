from interbotix_xs_modules.xs_launch import (
    declare_interbotix_xsarm_robot_description_launch_arguments,
)
from interbotix_common_modules.launch import (
    AndCondition,
)
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

    # camera setting
    rs_actions = []
    camera_names = [
        LaunchConfiguration('cam_high_name'),
        LaunchConfiguration('cam_left_wrist_name'),
        LaunchConfiguration('cam_right_wrist_name')
    ]
    for camera_name in camera_names:
        rs_actions.append(
            Node(
                package='realsense2_camera',
                namespace=camera_name,
                name='camera',
                executable='realsense2_camera_node',
                parameters=[
                    {'initial_reset': True},
                    ParameterFile(
                        param_file=PathJoinSubstitution([
                            FindPackageShare('meloha'),
                            'config',
                            'rs_cam.yaml',
                        ]),
                        allow_substs=True,
                    )
                ],
                output='screen',
            ),
        )

    realsense_ros_launch_includes_group_action = GroupAction(
      condition=IfCondition(LaunchConfiguration('use_cameras')),
      actions=rs_actions,
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up ALOHA with the following launch configurations: ',
        '\n- launch_leaders: ', LaunchConfiguration('launch_leaders'),
        '\n- use_cameras: ', LaunchConfiguration('use_cameras'),
    ])

    return [
        realsense_ros_launch_includes_group_action,
        loginfo_action,
    ]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_leader_left',
            default_value='leader_left',
            description='name of the left leader arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_leader_right',
            default_value='leader_right',
            description='name of the right leader arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_follower_left',
            default_value='follower_left',
            description='name of the left follower arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_name_follower_right',
            default_value='follower_right',
            description='name of the right follower arm',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_modes_left',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'leader_modes_left.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the left leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_modes_right',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'leader_modes_right.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the right leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'follower_modes_left',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'follower_modes_left.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the left follower arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'follower_modes_right',
            default_value=PathJoinSubstitution([
                FindPackageShare('aloha'),
                'config',
                'follower_modes_right.yaml',
            ]),
            description="the file path to the 'mode config' YAML file for the right follower arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'launch_leaders',
            default_value='true',
            choices=('true', 'false'),
            description=(
                'if `true`, launches both the leader and follower arms; if `false, just the '
                'followers are launched'
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_cameras',
            default_value='true',
            choices=('true', 'false'),
            description='if `true`, launches the camera drivers.',
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
    
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_motor_specs_left',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'leader_motor_specs_left.yaml'])
            ],
            description="the file path to the 'motor specs' YAML file for the left leader arm.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'leader_motor_specs_right',
            default_value=[
                PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'leader_motor_specs_right.yaml'])
            ],
            description="the file path to the 'motor specs' YAML file for the right leader arm.",
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
