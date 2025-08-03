from collections import deque
import time
from typing import Sequence, Union
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import os
import cv2
import math

from meloha.constants import (
    DATA_DIR,
    COLOR_IMAGE_TOPIC_NAME,
    DT,
    LEFT_ARM_START_POSE,
    RIGHT_ARM_START_POSE
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

        # Member variable for calculating position displacement
        self.initial_position: np.ndarray = None
        self.current_position: np.ndarray = None
        self.displacement: np.ndarray = None
        self.button: bool = False # Whether the Vive Tracker button has been pressed

        # VIVE Tracker Module Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.target_frame = self.node.declare_parameter(
            f'{side}_target_frame', 'vive_world').get_parameter_value().string_value
        self.source_frame = self.node.declare_parameter(
            f'{side}_source_frame', tracker_sn).get_parameter_value().string_value
        
        # Subscribe to button press events to control joint state publishing
        self.joy_subscriber = self.node.create_subscription(
            Joy,
            'libsurvive/joy',
            self.joy_callback,
            10
        )

        self.timer = self.node.create_timer(DT, self.get_tracker_position)

        if self.is_debug:
            self.displacement_history = []

        while self.initial_position is None and rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.get_logger().info(f'Found VIVE Tracker {self.side} position. Continuing...')    
        self.node.get_logger().info(f"VIVE Tracker {self.side} is connected well!")

    def get_tracker_position(self):

        """ 
        Retrieve the current position of the VIVE Tracker from the TF tree
        """

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

        if self.initial_position is None:
            self.initial_position = np.array([pos.x, pos.y, pos.z])
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.displacement = np.array([0.0, 0.0 ,0.0])
        else:
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.displacement = self.current_position - self.initial_position

        # TODO : vive tracker의 x,y,z 변위를 변수에다가 모두 저장하도록 코드 구성하기
        if self.is_debug:
            self.displacement_history.append([pos.x, pos.y, pos.z])

        return


    def joy_callback(self, msg: Joy):
        # 버튼 눌림 상태 확인
        tracker_sn:str = msg.header.frame_id
        button_states = msg.buttons

        if any(button_states):

            if tracker_sn == "LHR-21700E73":
                self.button = not self.button
                self.node.get_logger().info(f"Left Vive Tracker Button Pressed")
            elif tracker_sn == "LHR-0B6AA285":
                self.button = not self.button
                self.node.get_logger().info(f"Right Vive Tracker Button Pressed")

    def print_diagnostics(self):
        def dt_helper(ts):
            ts = np.array(ts)
            diff = ts[1:] - ts[:-1]
            return np.mean(diff)

        tracker_pub_freq = 1 / dt_helper(self.tracker_pub_timestamps)
        print(f'{tracker_pub_freq=:.2f}')

        
def get_arm_joint_positions(bot: Manipulator) -> list:
    return bot.joint_states


def move_arms(
    bot_list: Sequence[Manipulator],
    target_pose_list: Sequence[Sequence[float]],
    moving_time: float = 1.0
)-> None:
    
    num_steps = int(moving_time/DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.set_joint_positions(traj_list[bot_id][t])
        time.sleep(DT)

def move_arms_sim(
    bot_list: Sequence[Manipulator],
    target_pose_list: Sequence[Sequence[float]],
    moving_time: float = 1.0
)-> None:
    
    num_steps = int(moving_time/DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    zipped_lists = zip(curr_pose_list, target_pose_list)
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zipped_lists
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.set_joint_positions(traj_list[bot_id][t])
        time.sleep(DT)

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
        For testing if vive tracker 3.0 is communicated well.
    """

    node = create_meloha_global_node('meloha')
    vive_tracker = ViveTracker(
        side='left',
        tracker_sn='LHR-21700E73',
        node=node,
        is_debug=True,
        )

    # start up global node
    robot_startup(node)

    node.get_logger().info("vive tracker is started!")

    node.get_logger().info(f"wait for 5 seconds")
    time.sleep(5)

    start_time = time.time()
    while time.time() - start_time < 5:
        time.sleep(1.0/30.0)

    plot_vive_tracker_displacement(vive_tracker.displacement_history)

    robot_shutdown(node)


def plot_vive_tracker_displacement(displacement_history):
    """
    Plot the x, y, and z components of the Vive Tracker's displacement over time.
    Each axis subplot has its own autoscaled y-limits (based on data),
    but uses a consistent tick interval of 0.2.
    """

    if not displacement_history:
        print("No displacement data available to plot.")
        return

    displacement_array = np.array(displacement_history)
    time_steps = np.arange(len(displacement_array)) / 30.0  # Assume 30 Hz

    axis_labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    plt.figure(figsize=(10, 8))
    for i in range(3):
        values = displacement_array[:, i]
        min_val = np.min(values)
        max_val = np.max(values)

        # y축 범위를 0.2 간격으로 맞춰서 딱 떨어지게
        y_min = math.floor(min_val * 5) / 5.0  # 0.2 단위로 내림
        y_max = math.ceil(max_val * 5) / 5.0   # 0.2 단위로 올림

        # 데이터가 너무 평평할 경우 중앙 정렬을 위해 여유를 줌
        if y_max - y_min < 0.4:
            center = (y_min + y_max) / 2
            y_min = center - 0.2
            y_max = center + 0.2

        plt.subplot(3, 1, i + 1)
        plt.plot(time_steps, values, color=colors[i], label=f'{axis_labels[i]} Displacement')
        plt.ylabel(f'{axis_labels[i]} (meters)')
        plt.ylim(y_min, y_max)
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))  # 눈금 간격 고정
        plt.grid(True)
        plt.legend(loc='upper right')
        if i == 2:
            plt.xlabel("Time (seconds)")

    plt.suptitle("Vive Tracker Displacement Over Time (per axis)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def test_move_arms():

    node = create_meloha_global_node('meloha')

    follower_bot_left = Manipulator(
        side="left",
        node=node,
    )
    follower_bot_right = Manipulator(
        side="right",
        node=node,
    )    

    target_pose = [1.0, 0.3, 0.1]
    node.get_logger().info(f"박스를 잡는 위치로 이동합니다")
    move_arms_sim(
        [follower_bot_left, follower_bot_right],
        [target_pose, -target_pose],
        moving_time=2
    )
    node.get_logger().info(f"이동을 마쳤습니다.")

    time.sleep(1)

    node.get_logger().info(f"home position으로 이동합니다.")
    move_arms_sim(
        [follower_bot_left, follower_bot_right],
        [convert_angle_to_position(LEFT_ARM_START_POSE), convert_angle_to_position(RIGHT_ARM_START_POSE)],
        moving_time=2
    )
    node.get_logger().info(f"이동을 마쳤습니다.")

if __name__ == "__main__":
    test_vive_tracker()