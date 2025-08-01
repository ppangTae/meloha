from typing import Tuple, Dict, List, Optional, Union

import os
import yaml
import numpy as np
import time
import math
from threading import Lock
from pathlib import Path

from rclpy.node import Node
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory

from meloha.robot import (
    MelohaRobotNode,
    create_meloha_global_node,
    robot_startup,
    robot_shutdown,
)

from meloha.utils import get_transformation_matrix

from meloha.constants import MOTOR_ID

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from multi_dynamixel_interfaces.msg import MultiSetPosition


class Manipulator:

    def __init__(
        self,
        side: str,
        is_debug: bool = False,
        node: Node = None,
    ):

        # robot configuation
        self.side = side
        self.ns = f"follower_{self.side}"  # name space
        self.motor_id = MOTOR_ID[self.side]
        self.dh_param: dict = self._load_dh_params(self.side)

        self.T10 = (
            self.get_T10()
        )  # Transformation matrix from joint 1 to base frame (joint 0)

        # self.joint_states: list = None # -> real robot

        # simulation 
        if side == 'left':
            self.joint_states: list = [0.0, -0.6259112759506524, 0.6259050168378929]
        elif side == 'right':
            self.joint_states: list = [0.0, 0.6259112759506524, -0.6259112759506524]
        self.current_ee_position = None

        self.js_mutex = Lock()

        if node is None:
            self.node = create_meloha_global_node("meloha")
        else:
            self.node = node

        cb_group_manipulator = ReentrantCallbackGroup()

        self.pub_single = self.node.create_publisher(
            msg_type=SetPosition,
            topic="/set_position",
            qos_profile=10,
            callback_group=cb_group_manipulator,
        )
        node.get_logger().info(
            f"Manipulator follower {side} single joint command publisher is created!"
        )

        self.pub_group = self.node.create_publisher(
            msg_type=MultiSetPosition,
            topic="/multi_set_position",
            qos_profile=10,
            callback_group=cb_group_manipulator,
        )
        node.get_logger().info(
            f"Manipulator follower {side} joint commands publisher is created!"
        )

        self.sub_joint_states = self.node.create_subscription(
            msg_type=JointState,
            topic=f"{self.ns}/joint_states",
            callback=self._joint_state_cb,
            qos_profile=10,
            callback_group=cb_group_manipulator,
        )
        node.get_logger().info(
            f"Manipulator follower {side} joint states subscriber is created!"
        )

        # Find current joint_positions and ee position
        self.node.get_logger().debug(
            f'Trying to find joint states on topic "follwer_{side}/joint_states"...'
        )
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.get_logger().debug("Found joint states. Continuing...")
        self.current_ee_position = self._solve_fk(self.joint_states)
        node.get_logger().info(
            f"Manipurator {self.side} is located in {self.current_ee_position}"
        )
        node.get_logger().info(
            f"Manipurator {self.side} is located in {self.joint_states}"
        )
        node.get_logger().info(f"Manipulator {side} is created well!")

    def _joint_state_cb(self, msg: JointState):

        with self.js_mutex:
            self.joint_states = msg.position

        # Joint state에 맞춰 fk계산해서 current_position update
        self.current_ee_position = self._solve_fk(self.joint_states)

    def _load_dh_params(self, side):
        dh_param_filename = f"{side}_dh_param.yaml"
        package_share = Path(get_package_share_directory("meloha"))
        yaml_file_path = package_share / "config" / dh_param_filename
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        return data["dh_params"]

    def solve_ik(self, target: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Solves the inverse kinematics (IK) for a 3-DOF robotic arm using an algebraic approach,
        specifically selecting the "upper arm" (elbow-up) solution. The target position is given
        with respect to joint0 (base), but for IK calculation, it is transformed to the joint1 frame.
        This transformation is necessary because the algebraic solution requires the end-effector
        position relative to joint1. Computes joint angles (theta1, theta2, theta3) and returns
        whether a valid solution exists along with the joint values.

        Args:
            target (np.ndarray): 3D target position in the base frame.

        Returns:
            Tuple[bool, np.ndarray]: (success flag, joint angles array)
        """

        try:
            target = np.append(target, 1)
            target_in_joint1_frame = self.T10 @ target
            target_x, target_y, target_z, _ = target_in_joint1_frame

            # 링크 길이
            L1 = self.dh_param["joint1"]["d"]
            L2 = self.dh_param["joint2"]["a"]
            L3 = self.dh_param["joint3"]["a"]

            A = np.linalg.norm([target_x, target_y])
            B = target_z - L1

            # theta1
            theta1 = math.atan2(target_y, target_x)

            # theta3
            c3 = (A**2 + B**2 - (L2**2 + L3**2)) / (2*L2 *L3)
            if abs(c3) > 1.0:
                raise ValueError(f"Invalid value for c3={c3}: outside of [-1, 1]")

            sign = 1 if self.side == "left" else -1
            s3 = sign * np.sqrt(1 - c3**2)
            theta3 = math.atan2(s3, c3)

            # theta2
            if self.side == "left":
                s2 = (1 / (A**2 + B**2)) * (-L3*s3*A - (L2+L3*c3)*B)
                c2 = (1 / (A**2 + B**2)) * (-L3*s3*B + (L2+L3*c3)*A)
            elif self.side == "right":
                s2 = -(1 / (A**2 + B**2)) * (L3*s3*A - (L2+L3*c3)*B)
                c2 = -(1 / (A**2 + B**2)) * (-(L2+L3*c3)*A - L3*s3*B)
            theta2 = math.atan2(s2, c2)

            # 결과 조인트 각도
            js_target = np.array([theta1, theta2, theta3])

            if not np.isfinite(js_target).all():
                raise ValueError("NaN or Inf detected in joint solution")

            return True, js_target

        except Exception as e:
            self.node.get_logger().error(f"IK computation fail : {e}")

            return False, self.joint_states

    def _solve_fk(self, positions: np.ndarray) -> np.ndarray:
        """
        Computes the forward kinematics (FK) for the manipulator using Denavit-Hartenberg parameters.

        Args:
            positions (np.ndarray): The joint positions (angles) for the manipulator.

        Returns:
            np.ndarray: The 3D position of the end-effector in the base frame.
        """

        T = np.eye(4)
        pos_idx = 0

        # Use the order of keys as they appear in the loaded DH param dict
        for joint, param in self.dh_param.items():
            theta = param.get("theta")
            # For actuated joints, override theta with input positions
            if "theta" not in param and pos_idx < len(positions):
                theta = positions[pos_idx]
                pos_idx += 1
            alpha = param.get("alpha")
            a = param.get("a")
            d = param.get("d")
            A = get_transformation_matrix(theta, alpha, a, d)
            T = T @ A

        return T[:3, 3]  # End-effector position in base frame

    def get_T10(self):
        """
        Returns the transformation matrix from joint 1 to the base frame (joint 0).
        This is a static transformation based on the DH parameters.

        """

        T0_05 = get_transformation_matrix(
            theta=self.dh_param["joint0"]["theta"],
            alpha=self.dh_param["joint0"]["alpha"],
            a=self.dh_param["joint0"]["a"],
            d=self.dh_param["joint0"]["d"],
        )
        T05_1 = get_transformation_matrix(
            theta=self.dh_param["joint0.5"]["theta"],
            alpha=self.dh_param["joint0.5"]["alpha"],
            a=self.dh_param["joint0.5"]["a"],
            d=self.dh_param["joint0.5"]["d"],
        )
        T01 = T0_05 @ T05_1
        T10 = np.linalg.inv(T01)  # Cache the result

        return T10  # Inverse transformation from joint 1 to base frame (joint 0)

    def set_single_joint_position(
        self,
        motor_id: float,
        position: float,
    ):

        self.node.get_logger().debug(
            f"Setting Joint '{motor_id}' to position={position}"
        )
        self._publish_command(position)

    def set_joint_positions(
        self,
        joint_positions: List[float],
    ) -> bool:

        self.node.get_logger().debug(f"Setting {joint_positions=}")
        if self._check_collision():
            self._publish_commands(joint_positions)
            return True
        else:
            return False

    def _publish_command(
        self,
        motor_id: float,
        position: float,
    ) -> None:

        self.node.get_logger().debug(
            f"Publishing Joint '{motor_id}' to position={position}"
        )
        from meloha.robot_utils import (
            convert_angle_to_position,
        )  # for solving circular import

        position: Union[List, float] = convert_angle_to_position(position)

        msg = SetPosition()
        msg.id = motor_id
        msg.position = position
        self.pub_single.publish(msg)

    def _publish_commands(
        self,
        positions: List[float],
    ) -> None:

        from meloha.robot_utils import (
            convert_angle_to_position,
        )  # for solving circular import

        positions: Union[List, float] = convert_angle_to_position(positions)
        self.node.get_logger().debug(f"Publishing {positions=}")

        msg = MultiSetPosition()
        msg.ids = self.motor_id
        msg.positions = positions

        self.pub_group.publish(msg)

    def _check_collision(self):

        # Check collision
        x, y, z = self.current_ee_position

        if (-0.18 <= x <= 0.18) and (0 <= y <= 0.6) and (-0.12 <= z <= 0.12):
            self.node.get_logger().error(
                f"[충돌 위험] End-effector가 몸체에 부딪힐 수 있습니다: x={x:.2f}, y={y:.2f}, z={z:.2f}\n \
                    안전을 위해 프로그램을 강제종료 합니다."
            )
            rclpy.shutdown()
        elif (-0.24 <= x <= 0.24) and (y <= -0.60) and (-0.24 <= z <= 0.24):
            self.node.get_logger().error(
                f"[충돌 위험] End-effector가 주행부에 부딪힐 수 있습니다: x={x:.2f}, y={y:.2f}, z={z:.2f}\n \
                    안전을 위해 프로그램을 강제종료 합니다."
            )
            rclpy.shutdown()
        return True

    def get_node(self) -> MelohaRobotNode:
        return self.node


def calculate_ik_computation_time():

    node = create_meloha_global_node("meloha")

    left_manipulator = Manipulator(side="left", node=node)

    robot_startup(node)

    workspace_dir = os.path.expanduser("~/meloha_ws")
    file_path = os.path.join(workspace_dir, "src/meloha/joint_data/displacement.csv")
    displacements = np.loadtxt(file_path, delimiter=",", skiprows=1)

    left_disp = displacements[1:, :3]  # 160x3
    durations = []

    # 초기 위치 설정
    current_pos = left_manipulator.current_ee_position

    for disp in left_disp:
        target_position = current_pos + disp

        start = time.perf_counter()
        left_joint_angles = left_manipulator.solve_ik(target_position)
        end = time.perf_counter()

        left_manipulator.joint_states = left_joint_angles
        left_manipulator.current_ee_position = target_position

        durations.append(end - start)

    avg_time_ms = np.mean(durations) * 1000
    print(f"총 {len(left_disp)}개의 IK 평균 계산 시간: {avg_time_ms:.4f} ms")


def check_collsion_function():

    node = create_meloha_global_node("meloha")

    left_manipulator = Manipulator(side="left", node=node)

    robot_startup(node)

    collision_target = [180, -100, -400]
    left_manipulator.current_ee_position = collision_target
    left_joint_angles = left_manipulator.solve_ik(collision_target)
    left_manipulator.set_joint_positions(left_joint_angles)


if __name__ == "__main__":
    check_collsion_function()
