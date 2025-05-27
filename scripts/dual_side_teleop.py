#!/usr/bin/env python3
import numpy as np
import argparse
import signal
import time
import math
from functools import partial
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from multi_dynamixel_interfaces.msg import MultiSetPosition
from meloha.manipulator import (
    Manipulator
)

from meloha.robot_utils import (
    ViveTracker,
    convert_angle_to_position,
)

from meloha.robot import (
    create_meloha_global_node,
    robot_shutdown,
    robot_startup,
)
import rclpy

def main(args) -> None:

    # time.sleep(5)

    node = create_meloha_global_node('meloha')

    is_sim = args['is_sim']
    is_sim = False
    if is_sim:
        print("Running in simulation mode (RViz).")
        joint_state_publisher = node.create_publisher(JointState, 'joint_states', 10)

    follower_bot_left = Manipulator(
        side='left',
        node=node,
    )
    follower_bot_right = Manipulator(
        side='right',
        node=node,
    )
    tracker_left = ViveTracker(
        side='left',
        tracker_sn='LHR-21700E73',
        node=node,
    )
    tracker_right = ViveTracker(
        side='right',
        tracker_sn='LHR-0B6AA285',
        node=node
    )

    robot_startup(node)

    # follower_bot_left.go_to_home_pose()
    # follower_bot_right.go_to_home_pose()

    while rclpy.ok():
        displacement = 1.0 * np.array([tracker_left.displacement, tracker_right.displacement])
        displacement[0][0] = displacement[0][0] * (-1) # left x축 방향 조정
        displacement[1][0] = displacement[1][0] * (-1)
        if tracker_left.update_disp: # Pressed left button, then start to move
            ee_target = np.array([follower_bot_left.current_ee_position + displacement[0],
                                  follower_bot_right.current_ee_position + displacement[1]])
            left_ik_success, left_action = follower_bot_left.solve_ik(ee_target[0])
            right_ik_success, right_action = follower_bot_right.solve_ik(ee_target[1])
            action = np.concatenate([left_action, right_action])
            if is_sim:
                js = JointState()
                js.header = Header()
                js.header.stamp = node.get_clock().now().to_msg()
                js.name = ['joint_1_left', 'joint_2_left', 'joint_3_left',
                           'joint_1_right', 'joint_2_right', 'joint_3_right']
                js.position = list(action)
                joint_state_publisher.publish(js)
                if left_ik_success:
                    follower_bot_left.current_ee_position = ee_target[0]
                if right_ik_success:
                    follower_bot_right.current_ee_position = ee_target[1]
                node.get_logger().info(f'JointState 발행: {js.position}')
            else:
                action = list(action)
                follower_bot_left.set_joint_positions(action[:3])
                follower_bot_right.set_joint_positions(action[3:])
                
                if left_ik_success:
                    follower_bot_left.current_ee_position = ee_target[0]
                if right_ik_success:
                    follower_bot_right.current_ee_position = ee_target[1]

        time.sleep(1.0 / 50)
    robot_shutdown(node)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_sim',
        action='store',
        default=True,
        help='If set, runs in RViz (simulation mode). Otherwise, runs on real robot.'
    )
    args = parser.parse_args()
    main(vars(args))  # 딕셔너리 형태로 main에 전달
