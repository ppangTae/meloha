
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
    # dynamixel_read_write_node = Node(
    #     package="dynamixel_sdk_examples",
    #     executable="read_write_node",
    #     namespace="dynamixel",
    #     name="dynamixel"
    # )

    # Include external launch file from libsurvive_ros2 package
    libsurvive_include_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('libsurvive_ros2'),
                'launch',
                'libsurvive_ros2.launch.py'
            ])
        ),
    )

    rviz_include_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('meloha'),
                'launch',
                'meloha_rviz.launch.py'
            ])
        ),
    )

    usb_cam_head = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        namespace='usb_cam_head',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('meloha'),
                'config',
                'usb_cam_head.yaml',
            ]),
        ]
    )

    # usb_cam_high = Node(
    #     package='usb_cam',
    #     executable='usb_cam_node_exe',
    #     name='camera',
    #     namespace='usb_cam_high',
    #     output='screen',
    #     parameters=[
    #         PathJoinSubstitution([
    #             FindPackageShare('meloha'),
    #             'config',
    #             'usb_cam_high.yaml',
    #         ]),
    #     ]
    # )

    usb_cam_left_wrist = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        namespace='usb_cam_left_wrist',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('meloha'),
                'config',
                'usb_cam_left_wrist.yaml',
            ]),
        ]
    )

    usb_cam_right_wrist = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        namespace='usb_cam_right_wrist',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('meloha'),
                'config',
                'usb_cam_right_wrist.yaml',
            ]),
        ]
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up MELOHA with the following launch configurations: ',

    ])

    return [
        #dynamixel_read_write_node,
        libsurvive_include_launch,
        rviz_include_launch,
        # usb_cam_high,
        usb_cam_head,
        usb_cam_left_wrist,
        usb_cam_right_wrist,
        loginfo_action,
    ]


def generate_launch_description():
    declared_arguments = []

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
