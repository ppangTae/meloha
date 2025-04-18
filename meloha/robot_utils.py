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
from meloha.vive_tracker import ViveTrackerModule, vr_tracked_device

from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped

class ImageRecorder:

    def __init__(
        self,
        is_debug: bool = False,
        node: Node = None,
    ):
        self.is_debug = is_debug
        self.node = node
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
            self.node.create_subscription(Image, topic, callback_func, 20)

            self.node.get_logger().info(f"{cam_name} ImageRecorder Subscriber is created.")
            self.node.get_logger().info(f" Topic name : {topic}\n")

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
        self.node.get_logger().debug(f'{cam_name}_image is updated!')

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
        self.qpos = None
        self.arm_command = None
        self.is_debug = is_debug
        self.node = node

        self.node.create_subscription(
            JointState,
            f'/follower_{side}/joint_states',
            self.follower_state_cb,
            10,
        )
        self.node.get_logger().info(f"JointState Subscriber is created!")

        # node.create_subscription(
        #     JointGroupCommand,
        #     f'/follower_{side}/commands/joint_group',
        #     self.follower_arm_commands_cb,
        #     10,
        # )

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)


    # def follower_arm_commands_cb(self, data: JointGroupCommand):
    #     self.arm_command = data.cmd
    #     if self.is_debug:
    #         self.arm_command_timestamps.append(time.time())

    def follower_state_cb(self, data: JointState):
        self.qpos = data.position
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())
        self.node.get_logger().debug("JointStates is updated")

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n')


class ViveTracker:
    def __init__(
        self,
        is_debug: bool = False,
        node: Node = None,
    ):
        self.is_debug = is_debug
        self.node = node

        # VIVE Tracker 모듈 초기화
        self.vive_tracker = ViveTrackerModule()
        self.vive_tracker.print_discovered_objects()

        # 디바이스 할당
        tracker_1: vr_tracked_device = self.vive_tracker.devices.get("tracker_1")
        tracker_2: vr_tracked_device = self.vive_tracker.devices.get("tracker_2")

        # 시리얼 넘버 읽기
        tracker_1_serial_number = tracker_1.get_serial()
        tracker_2_serial_number = tracker_2.get_serial()
        self.node.get_logger().debug(f"tracker1 founded : {tracker_1_serial_number}")
        self.node.get_logger().debug(f"tracker2 founded : {tracker_2_serial_number}")

        # 미리 지정된 시리얼 넘버 (A: left, B: right)
        LEFT_TRACKER_SERIAL = "A"
        RIGHT_TRACKER_SERIAL = "B" 

        # 트래커 배정
        if tracker_1_serial_number == LEFT_TRACKER_SERIAL:
            self.tracker_left = tracker_1
            self.tracker_right = tracker_2
        elif tracker_1_serial_number == RIGHT_TRACKER_SERIAL:
            self.tracker_left = tracker_2
            self.tracker_right = tracker_1
        else:
            raise ValueError("Tracker serial numbers do not match expected values.")


        if not self.tracker_left or not self.tracker_right:
            raise Exception("Trackers not found properly.")

        # position 변위를 게산하기 위한 멤버변수
        self.previous_left_position: list = self.tracker_left.get_pose_euler() # [x, y, z, yaw, pitch, roll]
        self.current_left_position = None
        self.previous_right_position: list = self.tracker_right.get_pose_euler()
        self.current_right_position = None

        self.left_arm_displacement_publisher = node.create_publisher(PoseStamped, '/follower_left/displacement', 10)
        self.right_arm_displacement_publisher = node.create_publisher(PoseStamped, '/follower_right/displacement', 10)

        # 1.0/30초 (약 33ms) 주기로 update_tracker_position 호출하는 Timer 설정
        self.node.create_timer(1.0 / 30.0, self.update_tracker_position)

        self.node.get_logger().info("VIVE Tracker is connected well!")

    def update_tracker_position(self):

        self.previous_left_position = self.current_left_position
        self.previous_right_position = self.current_right_position

        self.current_left_position = self.tracker_left.get_pose_euler()
        self.current_right_position = self.tracker_right.get_pose_euler()

        if self.is_debug:
            self.node.get_logger().debug(f"[ViveTracker] Updated right pose: {self.current_right_position}")

        if self.is_debug:
            self.node.get_logger().info(f"[ViveTracker] Updated right pose: {self.current_left_position}")
        
        self.publish_displacement()

    def publish_displacement(self):

        left_displacement = self.current_left_position - self.previous_left_position
        right_displacement = self.current_right_position - self.previous_right_position

        left_displacement_msg = PoseStamped()
        right_displacement_msg = PoseStamped()

        now = self.node.get_clock().now()

        left_displacement_msg.pose.position.x = left_displacement[0]
        left_displacement_msg.pose.position.y = left_displacement[1]
        left_displacement_msg.pose.position.z = left_displacement[2]
        left_displacement_msg.header.stamp = now.to_msg()

        right_displacement_msg.pose.position.x = right_displacement[0]
        right_displacement_msg.pose.position.y = right_displacement[1]
        right_displacement_msg.pose.position.z = right_displacement[2]
        right_displacement_msg.header.stamp = now.to_msg()

        self.left_arm_displacement_publisher.publish(left_displacement_msg)
        self.right_arm_displacement_publisher.publish(right_displacement_msg)

        
def get_arm_joint_positions(bot: Manipulator):
    return bot.joint_states.position

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
            bot.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


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

def test_vive_tracker():

    """
        For testing if vive tracker3.0 is communicated well.
    """

    node = create_meloha_global_node('meloha')
    vive_tracker = ViveTracker(node=node)

    # start up global node
    robot_startup(node)

    # Wait for until Vive Tracker is ready!
    time.sleep(5)

    node.get_logger().info("vive tracker is started!")

    while True:
        node.get_logger().info(f"left_pose : {vive_tracker.left_pose}")
        node.get_logger().info(f"right_pose : {vive_tracker.right_pose}")

        time.sleep(1.0/30.0)

    robot_shutdown(node)

if __name__ == "__main__":
    test_image_recorder()