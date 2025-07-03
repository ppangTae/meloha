#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


# ─────────────────────────── launch_setup ────────────────────────────
def launch_setup(context, *args, **kwargs):

    urdf_file_name = 'simple_meloha.urdf.xml'
    urdf = os.path.join(
        get_package_share_directory('meloha'),
        'urdf',
        urdf_file_name,
    )
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_vive_tracker')),
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_jsp_gui')),
    )

    libsurvive_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('libsurvive_ros2'),
                'launch',
                'libsurvive_ros2.launch.py'
            ])
        ]),
        launch_arguments={
            'namespace': 'libsurvive',
            'record': 'false',
            'foxbridge' : 'false',
            'rosbridge' : 'false',
            'composable' : 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_vive_tracker')),
    )

    usbcam_actions = []
    usbcam_names = [
        LaunchConfiguration('usbcam_high_name'),
        LaunchConfiguration('usbcam_left_wrist_name'),
        LaunchConfiguration('usbcam_right_wrist_name')
    ]
    for usbcam_name in usbcam_names:
        usbcam_actions.append(
            Node(
                package='usb_cam',
                executable='usb_cam_node_exe',
                namespace=usbcam_name,
                name='camera',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('meloha'),
                        'config',
                        f'{usbcam_name.perform(context)}.yaml',
                    ]),
                ],
                output='screen',
            ),
        )

    usbcam_ros_launch_includes_group_action = GroupAction(
      condition=IfCondition(LaunchConfiguration('use_cameras')),
      actions=usbcam_actions,
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            LaunchConfiguration('meloha_rviz_config')
        ],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    loginfo_action = LogInfo(msg=[
         '\nBringing up MELOHA with the following launch configurations: ',
         '\n- use_vive_tracker: ', LaunchConfiguration('use_vive_tracker'),
         '\n- use_cameras: ', LaunchConfiguration('use_cameras'),
         '\n- use_rviz: ', LaunchConfiguration('use_rviz'),
         '\n- use_jsp_gui: ', LaunchConfiguration('use_jsp_gui'),
    ])

    return [
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz2_node,
        libsurvive_launch_include,
        usbcam_ros_launch_includes_group_action,
        loginfo_action,
    ]


# ───────────────── generate_launch_description ───────────────────────
def generate_launch_description():

    declared_arguments = []

    # About VIVE Tracker LaunchArgument
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_vive_tracker',
            default_value='False',
            choices=('True', 'False'),
            description='If True, launch libsurvive_ros2 for VIVE Tracker'
        )
    )

    # About Camera LaunchArgument
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_cameras',   
            default_value='False',
            choices=('True', 'False'),
            description='If True, launch USB camera nodes'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'usbcam_high_name',
            default_value='usb_cam_high',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'usbcam_left_wrist_name',
            default_value='usb_cam_left_wrist',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'usbcam_right_wrist_name',
            default_value='usb_cam_right_wrist',
        )
    )

    # About Rviz2 LaunchArgument
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_rviz',   
            default_value='False',
            choices=('True', 'False'),
            description='If True, launch rviz2 node')
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'meloha_rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('meloha'),
                'rviz',
                'meloha.rviz',
            ]),
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_jsp_gui',   
            default_value='True',
            choices=('True', 'False'),
            description='If True, launch joint_state_publisher_gui node')
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
