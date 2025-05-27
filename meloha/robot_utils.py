from collections import deque
import time
from typing import Sequence, Union

import os
import cv2
import math

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

import rclpy
from cv_bridge import CvBridge
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class ImageRecorder:

    def __init__(
        self,
        is_debug: bool = False,
        node: Node = None,
    ):
        self.is_debug = is_debug
        self.node = node
        self.bridge = CvBridge()

        self.camera_names = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
            elif cam_name == 'cam_head':
                callback_func = self.image_cb_cam_head
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
        time.sleep(0.1) # for stabilization

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
    
    def image_cb_cam_head(self, data):
        cam_name = 'cam_head'
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
        self.arm_command = None
        self.is_debug = is_debug
        self.node = node

        self.node.create_subscription(
            JointState,
            f'/follower_{side}/joint_states',
            self.follower_state_cb,
            10,
        )
        self.node.get_logger().info(f"recorder {side} is created!")

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def follower_state_cb(self, data: JointState):
        self.qpos = data.position
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())
        self.node.get_logger().debug(f"JointState positions subsribed : {self.qpos}")

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)

        print(f'{joint_freq=:.2f}')


"""
    https://github.com/asymingt/libsurvive_ros2
    You can utilize this library to get the location and orientation of the VIVE Tracker.
    This class checks the tf message sent by the above library to determine the location of the VIVE Tracker.

"""

class ViveTracker:
    def __init__(
        self,
        side: str,
        tracker_sn: str,
        is_debug: bool = False,
        node: Node = None,
    ):
        
        self.side = side
        self.tracker_sn = tracker_sn
        self.is_debug = is_debug
        self.node = node

        self.initialized = False

        # position 변위를 게산하기 위한 멤버변수
        self.previous_position: np.ndarray = None
        self.current_position: np.ndarray = None
        self.displacement: np.ndarray = None
        self.update_disp: bool = True # TODO : 바꿔야됨

        # VIVE Tracker 모듈 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # Serial Number of tracker
        self.target_frame = self.node.declare_parameter(
            f'{side}_target_frame', 'LHB-52B7DDF6').get_parameter_value().string_value
        self.source_frame = self.node.declare_parameter(
            f'{side}_source_frame', tracker_sn).get_parameter_value().string_value
        
        # button이 눌렸을 때는 joint state를 publish하지 않도록 하기 위해 button이 눌렸는지 안눌렸는지 정보를 받아오는 subscriber 생성
        self.joy_subscriber = self.node.create_subscription(
            Joy,
            'libsurvive/joy',  # 너가 설정한 joy_topic 이름이랑 같아야 함
            self.joy_callback,
            10
        )

        self.timer = self.node.create_timer(DT, self.get_tracker_disp_from_tf)

        while self.previous_position is None and rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.get_logger().debug(f'Found VIVE Tracker {self.side} position. Continuing...')    
        self.node.get_logger().info(f"VIVE Tracker {self.side} is connected well!")

    def get_tracker_disp_from_tf(self):

        try:
            tracker_tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.node.get_logger().info(
                f'Could not transform {self.target_frame} to {self.source_frame}({self.side}): {ex}'
            )
            return
        
        pos = tracker_tf.transform.translation

        if not self.initialized:
            self.previous_position = np.array([pos.x, pos.y, pos.z])
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.displacement = np.array([0.0, 0.0 ,0.0])
            self.initialized = True
        else:
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.displacement = self.current_position - self.previous_position
            self.previous_position = self.current_position
        return


    def joy_callback(self, msg: Joy):
        # 버튼 눌림 상태 확인
        button_states = msg.buttons
        if any(button_states):
            self.update_disp = not self.update_disp
            self.node.get_logger().info(f'🔴 Button Pressed')
            if self.update_disp:
                self.node.get_logger().info(f"start to publish JointState")
            else:
                self.node.get_logger().info(f"stop to publish JointState")

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)

        tracker_pub_freq = 1 / dt_helper(self.tracker_pub_timestamps)
        print(f'{tracker_pub_freq=:.2f}')

        
def get_arm_joint_positions(bot: Manipulator):
    return bot.joint_states

def convert_angle_to_position(rad: Union[float, list]) -> Union[int, list]:
    """
        Convert radian angle(s) to ROBOTIS MOTOR Command Value(s)
        -pi -> -501923, pi -> 501923
        Supports scalar, list, or numpy array input.
    """
    max_input = math.pi
    max_output = 501923

    rad = np.asarray(rad)
    pos = (rad / max_input) * max_output
    result = pos.astype(int)
    
    if result.ndim == 0:
        return int(result)  # scalar
    else:
        return result.tolist()  # list of ints

def convert_position_to_angle(pos):
    """
    Convert ROBOTIS MOTOR Command Value(s) to radian angle(s)
    -501923 -> -pi, 501923 -> pi
    Supports scalar, list, or numpy array input.
    Always returns float or list[float].
    """
    max_input = 501923
    max_output = math.pi

    pos = np.asarray(pos)
    angle = (pos / max_input) * max_output

    if angle.ndim == 0:
        return float(angle)  # scalar
    else:
        return angle.tolist()  # list of floats


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

def test_joint_recorder():

    node = create_meloha_global_node('meloha')
    recorder_left = Recorder('left', node=node)
    robot_startup(node)
    node.get_logger().info("image_recorder is started!")
    while True:
        node.get_logger().info(f"left_displacement : {recorder_left.qpos}")

        time.sleep(1.0/30.0)
    return

def test_vive_tracker():

    """
        For testing if vive tracker3.0 is communicated well.
    """

    node = create_meloha_global_node('meloha')
    vive_tracker = ViveTracker(
        side='left',
        tracker_sn='LHR-21700E73',
        node=node)

    # start up global node
    robot_startup(node)

    # Wait for until Vive Tracker is ready!
    time.sleep(5)

    node.get_logger().info("vive tracker is started!")

    while True:
        node.get_logger().info(f"left_displacement : {vive_tracker.displacement}")

        time.sleep(1.0/30.0)

    robot_shutdown(node)

if __name__ == "__main__":
    test_joint_recorder()