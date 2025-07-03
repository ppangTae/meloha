from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    LogInfo,
)
from launch.substitutions import LaunchConfiguration

# ① 런타임 단계: 여기서 값을 읽어야 한다
def launch_setup(context, *args, **kwargs):
    arg_val = LaunchConfiguration('launch_argument_test').perform(context)
    # Python print → 터미널 stdout, LogInfo → ros2 launch 로그
    print(f"[print] launch_argument_test = {arg_val}")
    return [
        LogInfo(msg=['[LogInfo] launch_argument_test: ', arg_val])
    ]

# ② 정적 단계: 인자 선언 + OpaqueFunction 등록
def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'launch_argument_test',
            default_value='hello launch file',
            description='launch_argument_test',
        )
    ]

    return LaunchDescription(declared_arguments + [
        OpaqueFunction(function=launch_setup)  # ← 반드시 포함
    ])
