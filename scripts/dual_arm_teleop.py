#!/usr/bin/env python3
import numpy as np
import argparse
import signal
import time
import math
import pyfiglet
from functools import partial
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from multi_dynamixel_interfaces.msg import MultiSetPosition
from meloha.manipulator import (
    Manipulator
)

from meloha.constants import DT

from meloha.robot_utils import (
    ViveTracker,
    convert_angle_to_position,
)

from meloha.robot import (
    create_meloha_global_node,
    robot_shutdown,
    robot_startup,
)
from meloha.utils import print_countdown
import rclpy

def main(args) -> None:

    print_countdown("TELEOP START")

    node = create_meloha_global_node('meloha')

    is_sim = args['is_sim']
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
        moving_scale = 2.5
        left_displacement = moving_scale * tracker_left.displacement
        right_displacement = moving_scale * tracker_right.displacement

        left_displacement[1] = -left_displacement[1]  # Y-axis inversion for left side
        right_displacement[1] = -right_displacement[1]  # Y-axis inversion for right side

        if tracker_left.button: # Pressed left button, then start to move
            left_ee_target = follower_bot_left.current_ee_position + left_displacement
            right_ee_target = follower_bot_right.current_ee_position + right_displacement
            left_ik_success, left_action = follower_bot_left.solve_ik(left_ee_target)
            right_ik_success, right_action = follower_bot_right.solve_ik(right_ee_target)
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
                    follower_bot_left.current_ee_position = left_ee_target
                if right_ik_success:
                    follower_bot_right.current_ee_position = right_ee_target
                node.get_logger().debug(f'JointState 발행: {js.position}')
            else:
                action = list(action)
                follower_bot_left.set_joint_positions(action[:3])
                follower_bot_right.set_joint_positions(action[3:])
                
                if left_ik_success:
                    follower_bot_left.current_ee_position = left_ee_target
                if right_ik_success:
                    follower_bot_right.current_ee_position = right_ee_target

        time.sleep(DT)

        if not tracker_right.button:
            break
    robot_shutdown(node)

if __name__ == '__main__':
    # python3 dual_side_teleop.py -> real robot
    # python3 dual_side_teleop.py --is_sim -> simulation 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--is_sim',
        action='store_const',
        const=True,
        default=False,
        help='If set, runs in simulation. Default runs on real robot.'
    )
    args = parser.parse_args()
    main(vars(args))
