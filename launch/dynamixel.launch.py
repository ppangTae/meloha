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

    # Dynamixel node setting
    dynamixel_node = Node(
        package="dynamixel_sdk_examples",
        executable="read_write_node",
        namespace="dynamixel",
        name="dynamixel"
    )

    loginfo_action = LogInfo(msg=[
        '\nBringing up ALOHA with the following launch configurations: ',
    ])

    return [
        dynamixel_node,
        loginfo_action,
    ]


def generate_launch_description():


    return LaunchDescription([OpaqueFunction(function=launch_setup)])
