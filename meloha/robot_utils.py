from collections import deque
import time
from typing import Sequence

import os
import cv2

from meloha.constants import (
    DATA_DIR,
    COLOR_IMAGE_TOPIC_NAME,
    DT,
)

from meloha.robot import (
    create_meloha_global_node,
    robot_startup,
    robot_shutdown,
)

from meloha.manipulator import Manipulator
# from vive_tracker.vive_tracker_node import ViveTrackerNode

from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

class ImageRecorder:

    def __init__(
        self,
        is_debug: bool = False,
        node: Node = None,
    ):
        self.is_debug = is_debug
        self.bridge = CvBridge()

        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            topic = COLOR_IMAGE_TOPIC_NAME.format(cam_name)
            print(topic)
            node.create_subscription(Image, topic, callback_func, 20)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name: str, data: Image):
        setattr(
            self,
            f'{cam_name}_image',
            self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        )
        setattr(self, f'{cam_name}_secs', data.header.stamp.sec)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nanosec)
        if self.is_debug:
            getattr(
                self,
                f'{cam_name}_timestamps'
            ).append(data.header.stamp.sec + data.header.stamp.sec * 1e-9)

    def image_cb_cam_high(self, data):
        cam_name = 'cam_high'
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = 'cam_left_wrist'
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = 'cam_right_wrist'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()

class Recorder:
    def __init__(
        self,
        side: str,
        is_debug: bool = False,
        node: Node = None,
    ):
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        node.create_subscription(
            JointState,
            f'/follower_{side}/joint_states',
            self.follower_state_cb,
            10,
        )
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def follower_state_cb(self, data: JointState):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n')

def get_arm_joint_positions(bot: Manipulator):
    return bot.arm.core.joint_states.position[:6]

def move_arms(
    bot_list: Sequence[Manipulator],
    target_pose_list: Sequence[Sequence[float]],
    moving_time: float = 1.0,
) -> None:
    num_steps = int(moving_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


def sleep_arms(
    bot_list: Sequence[Manipulator],
    moving_time: float = 5.0,
    home_first: bool = True,
) -> None:
    """Command given list of arms to their sleep poses, optionally to their home poses first.

    :param bot_list: List of bots to command to their sleep poses
    :param moving_time: Duration in seconds the movements should take, defaults to 5.0
    :param home_first: True to command the arms to their home poses first, defaults to True
    """
    if home_first:
        move_arms(
            bot_list,
            [[0.0, -0.96, 1.16, 0.0, -0.3, 0.0]] * len(bot_list),
            moving_time=moving_time
        )
    move_arms(
        bot_list,
        [bot.arm.group_info.joint_sleep_positions for bot in bot_list],
        moving_time=moving_time,
    )

def setup_follower_bot(bot: Manipulator):
    bot.core.robot_reboot_motors('single', 'gripper', True)
    bot.core.robot_set_operating_modes('group', 'arm', 'position')
    bot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    torque_on(bot)


def setup_leader_tracker(tracker: ViveTrackerNode):
    tracker.core.robot_set_operating_modes('group', 'arm', 'pwm')
    tracker.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    torque_off(tracker)


def set_standard_pid_gains(bot: Manipulator):
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 800)
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)


def set_low_pid_gains(bot: Manipulator):
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 100)
    bot.core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)


def torque_off(bot: Manipulator):
    bot.core.robot_torque_enable('group', 'arm', False)
    bot.core.robot_torque_enable('single', 'gripper', False)


def torque_on(bot: Manipulator):
    bot.core.robot_torque_enable('group', 'arm', True)
    bot.core.robot_torque_enable('single', 'gripper', True)


def test_image_recorder():

    """
        For testing if realsense2 publisher and ImageRecorder's subscriber is communicated well.
    """

    node = create_meloha_global_node('meloha')
    image_recorder = ImageRecorder(node=node)

    # start up global node
    robot_startup(node)

    # Wait for until ImageRecorder is ready!
    time.sleep(5)

    node.get_logger().info("image_recorder is started!")
    # capture 3 picture for demonstrating this node can subscribe picture

    test_data_dir = os.path.join(DATA_DIR, 'test_image_recorder')
    if not os.path.isdir(test_data_dir):
        os.makedirs(test_data_dir)

    images = image_recorder.get_images()

    # Save each image using cv2
    for cam_name in image_recorder.camera_names:
        image = images.get(cam_name)
        if image is not None:
            # Construct file path for saving the image (using PNG format)
            test_file_path = os.path.join(test_data_dir, f"{cam_name}.png")
            cv2.imwrite(test_file_path, image)
            print(f"Saved image for {cam_name} at {test_file_path}")
        else:
            print(f"No image captured for {cam_name}, skipping save.")  
    
    robot_shutdown(node)

# def test_joint_recorder():
#     if node is None:
#         node = get_meloha_global_node()
#     if node is None:
#         node = create_meloha_global_node('meloha')
#     joint_recorder = Recorder(node=node,)

if __name__ == "__main__":
    test_image_recorder()