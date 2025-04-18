
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

    # Dynamixel node setting
    dynamixel_read_write_node = Node(
        package="dynamixel_sdk_examples",
        executable="read_write_node",
        namespace="dynamixel",
        name="dynamixel"
    )

    usb_cam_high = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        namespace='usb_cam_high',
        # parameters=[
        #     ParameterFile(
        #         param_file=PathJoinSubstitution([
        #             FindPackageShare('meloha'),
        #             'config',
        #             'usb_cam_high.yaml',
        #         ]),
        #         allow_substs=True,
        #     )
        # ],
        output='screen',
    )

    # ! for testing
    image_relay_left = Node(
        package='image_transport',
        executable='republish',
        name='relay_left',
        arguments=['raw', 'raw'],
        remappings=[
            ('in', '/usb_cam_high/image_raw'),
            ('out', '/usb_cam_left_wrist/image_raw')
        ],
        output='screen'
    )
    # ! for testing
    image_relay_right = Node(
        package='image_transport',
        executable='republish',
        name='relay_right',
        arguments=['raw', 'raw'],
        remappings=[
            ('in', '/usb_cam_high/image_raw'),
            ('out', '/usb_cam_right_wrist/image_raw')
        ],
        output='screen'
    )

    usb_cam_left_wrist = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_left_wrist',
        namespace='usb_cam_left_wrist',
        parameters=[
            ParameterFile(
                param_file=PathJoinSubstitution([
                    FindPackageShare('meloha'),
                    'config',
                    'usb_cam_left_wrist.yaml',
                ]),
                allow_substs=True,
            )
        ],
        output='screen',
    )

    usb_cam_right_wrist = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_right_wrist',
        namespace='usb_cam_right_wrist',
        parameters=[
            ParameterFile(
                param_file=PathJoinSubstitution([
                    FindPackageShare('aloha'),
                    'config',
                    'usb_cam_right_wrist.yaml',
                ]),
                allow_substs=True,
            )
        ],
        output='screen',
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up MELOHA with the following launch configurations: ',

    ])

    return [
        #dynamixel_read_write_node,
        usb_cam_high,
        image_relay_left,
        image_relay_right,
        #usb_cam_left_wrist,
        #usb_cam_right_wrist,
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
