"""

Test suite for robot utils
This scripts contains unit tests for verifying the functionality of ViveTracker, ImageRecorder, Recorder class
It tests 

# 1. vive tracker의 변위정보가 30hz로 잘 전달되고있는지
# 2. camera의 image정보가 30hz로 잘 전달되고있는지
# 3. dynamixel의 motor joint정보가 30hz로 잘 전달되고있는지

"""
import os
import sys
import time
import unittest

import rclpy
import launch
import launch.actions
import launch_testing.actions
import launch_testing.markers

from ament_index_python.packages import get_package_share_directory

from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, JointState, Joy

from meloha.constants import COLOR_IMAGE_TOPIC_NAME
from meloha.robot import get_meloha_global_node, robot_shutdown

def _check_topic_hz(self, topic_name, msg_type, min_hz):
    stamps = []

    sub = self.node.create_subscription(
        msg_type, topic_name,
        lambda msg: stamps.append(self.node.get_clock().now().nanoseconds),
        10)

    end_time = time.time() + WINDOW_SEC
    while time.time() < end_time:
        rclpy.spin_once(self.node, timeout_sec=0.1)

    self.node.destroy_subscription(sub)

    self.assertGreaterEqual(len(stamps), 2,
        f'No traffic on {topic_name}')

    # 주파수 계산
    deltas = [(stamps[i+1]-stamps[i])/1e9
                for i in range(len(stamps)-1)]
    hz = 1.0 / (sum(deltas)/len(deltas))
    self.assertGreaterEqual(hz, min_hz,
        f'{topic_name} too slow: {hz:.1f} Hz < {min_hz} Hz')
    self.assertLessEqual(hz, TARGET_HZ + TOLERANCE_HZ,
        f'{topic_name} too fast? {hz:.1f} Hz')

def generate_test_description():
    bringup = launch.actions.ExecuteProcess(
        cmd=[sys.executable,
             os.path.join(get_package_share_directory('meloha'),
                          'test', 'test_process', 'data_process.py')],
        output='screen',
        name='data_process')

    return (
        launch.LaunchDescription([
            bringup,
            launch_testing.actions.ReadyToTest()
        ]),
        {'data_process': bringup}
    )

TARGET_HZ   = 30.0
TOLERANCE_HZ = 2.0
WINDOW_SEC   = 3.0          # 샘플링 시간

class TestDataFrequency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        robot_node = get_meloha_global_node()
        robot_shutdown(robot_node)
        rclpy.shutdown()
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # ---------- 테스트 ----------
    def test_image_topics(self):
        cam_names = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
        for cam_name in cam_names:
            topic = COLOR_IMAGE_TOPIC_NAME.format(cam_name)
            _check_topic_hz(topic, Image, TARGET_HZ - TOLERANCE_HZ)

    def test_joint_states(self):
        for side in ['left', 'right']:
            _check_topic_hz(f'/follower_{side}/joint_states',
                                 JointState, TARGET_HZ - TOLERANCE_HZ)

    def test_tf_stream(self):
        _check_topic_hz('/tf', TFMessage, TARGET_HZ - TOLERANCE_HZ)
