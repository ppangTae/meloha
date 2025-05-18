from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    Union
)

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

from meloha.robot import (
    MelohaRobotNode,
    create_meloha_global_node,
    robot_startup,
    robot_shutdown
)

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

        self.side = side
        self.is_debug = is_debug
        self.ns = f'follower_{self.side}'

        self.motor_id = MOTOR_ID[self.side]
        self.dh_param: dict = self._load_dh_params(self.side)

        self.joint_states: list = None # rad
        self.initial_states: list = None
        self.base_T = np.eye(4)

        self.target_ee_position = None
        self.current_ee_position = None
        self.displacement = None

        self.js_mutex = Lock()

        if node is None:
            self.node = create_meloha_global_node("meloha")
        else:
            self.node = node

        cb_group_manipulator = ReentrantCallbackGroup()

        # Dynamixel의 위치제어를 위한 publisher
        self.pub_single = self.node.create_publisher(
            msg_type=SetPosition,
            topic=f'/set_position',
            qos_profile=10,
            callback_group=cb_group_manipulator
        )
        node.get_logger().info(f"Manipulator follower {side} joint commands publisher is created!")

        self.pub_group = self.node.create_publisher(
            msg_type = MultiSetPosition,
            topic="/multi_set_position",
            qos_profile=10,
            callback_group=cb_group_manipulator
        )

        # Dynamixel에서 오는 joint state를 받기 위한 subscriber
        self.sub_joint_states = self.node.create_subscription(
            msg_type=JointState,
            topic=f'{self.ns}/joint_states', 
            callback=self._joint_state_cb,
            qos_profile=10,
            callback_group=cb_group_manipulator,
        )
        node.get_logger().info(f"Manipulator follower {side} joint states subscriber is created!")

        # Find current joint_positions and ee position
        self.node.get_logger().debug(
            f'Trying to find joint states on topic "follwer_{side}/joint_states"...'
        )
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.get_logger().debug('Found joint states. Continuing...')
        self.P = self._solve_fk(self.side, self.joint_states)
        self.current_ee_position = self.P[:,3]
        node.get_logger().info(f"Maniputor {self.side} is located in {self.current_ee_position}")
        node.get_logger().info(f"Manipulator {side} is created well!")
    
    def _joint_state_cb(self, msg: JointState):
        """
        Get the latest JointState message through a ROS Subscriber Callback.

        :param msg: JointState message
        """
        with self.js_mutex:
            self.joint_states = msg.position

    def _load_dh_params(self, side):
        yaml_file_name = f"{side}_dh_param.yaml"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        yaml_file_path = os.path.join(parent_dir, 'config', yaml_file_name)
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data['dh_params']
    
    def solve_ik(self, target: np.ndarray) -> Tuple[bool, np.ndarray]:
        try:
            T_head_to_joint1 = np.linalg.inv(self.base_T)

            target = np.append(target, 1)
            target_in_joint1_frame = T_head_to_joint1 @ target
            target_x, target_y, target_z, _ = target_in_joint1_frame

            # 링크 길이
            L1 = self.dh_param['joint1']['d']
            L2 = self.dh_param['joint2']['a']
            L3 = self.dh_param['joint3']['a']

            A = np.linalg.norm([target_x, target_y])
            B = target_z - L1

            # theta1
            theta1 = math.atan2(target_y, target_x)

            # theta3
            c3 = (A**2 + B**2 - (L2**2 + L3**2)) / (2 * L2 * L3)
            if abs(c3) > 1.0:
                raise ValueError(f"Invalid value for c3={c3}: outside of [-1, 1]")

            sign = 1 if self.side == 'left' else -1
            s3 = sign * np.sqrt(1 - c3**2)
            theta3 = math.atan2(s3, c3)

            # theta2
            if self.side == 'left':
                s2 = (1 / (A**2 + B**2)) * (-L3 * s3 * A - (L2 + L3 * c3) * B)
                c2 = (1 / (A**2 + B**2)) * (-L3 * s3 * B + (L2 + L3 * c3) * A)
            elif self.side == 'right': 
                s2 = -(1 / (A**2 + B**2)) * (L3 * s3 * A - (L2 + L3 * c3) * B)
                c2 = -(1 / (A**2 + B**2)) * (-(L2 + L3 * c3) * A - L3 * s3 * B)
            theta2 = math.atan2(s2, c2)

            # 결과 조인트 각도
            js_target = np.array([theta1, theta2, theta3])

            if not np.isfinite(js_target).all():
                raise ValueError("NaN or Inf detected in joint solution")

            self.joint_states = js_target
            return True, js_target

        except Exception as e:
            self.node.get_logger().error(f'IK computation fail : {e}')
            return False, np.array(self.joint_states)
        
    
    def _solve_fk(self, side: str, positions: np.ndarray):
        # 순방향 기구학 계산
        i = 3 # joint 수
        j = i + 1
        
        #변환 행렬 초기화
        A = np.zeros((4, 4, i))
        T = np.zeros((4, 4, j))
        P = np.zeros((3, j))
        R = np.zeros((3, 3, j))

        # 베이스 변환 행렬 설정
        if side == 'left':
            self.base_T = np.array([[0, 0, -1, -0.1],
                                    [-1, 0 , 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
        elif side == 'right':
            self.base_T = np.array([[0, 0, 1, 0.1],
                                    [-1, 0, 0, 0,],
                                    [0, -1, 0, 0,],
                                    [0 ,0, 0, 1]])
        
        T[:, :, 0] = self.base_T
        P[:, 0] = self.base_T[0:3, 3]

        #각 관절에 대한 변호나 행렬 계산
        for idx in range(i):
            theta = positions[idx]
            d = self.dh_param[f'joint{idx+1}']['d']
            alpha = self.dh_param[f'joint{idx+1}']['alpha']
            a = self.dh_param[f'joint{idx+1}']['a']
            ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
            A[:, :, idx] = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

            T[:, :, idx+1] = T[:, :, idx] @ A[:, :, idx]
            P[:, idx+1] = T[0:3, 3, idx+1]
            R[:, :, idx+1] = T[0:3, 0:3, idx+1]
        
        # P[:,1] : Base frame에서 본 2번조인트의 위치, P[:,3] = Base frame에서 본 end_effector의 위치
        return P
    
    # def go_to_home_pose(self):
    #     self._publish_commands(
    #         positions = START_ARM_POSE,
    #     )
    
    def set_single_joint_position(
        self,
        motor_id: float,
        position: float,
    ):

        self.node.get_logger().debug(f"Setting Joint '{motor_id}' to position={position}")
        self._publish_command(position)

    def set_joint_positions(
        self,
        joint_positions: List[float],
    ) -> bool:

        self.node.get_logger().debug(f'Setting {joint_positions=}')
        if self._check_collision(joint_positions):
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
        from meloha.robot_utils import convert_angle_to_position # for solving circular import
        position: Union[List, float] = convert_angle_to_position(position)        

        msg = SetPosition()
        msg.id = motor_id
        msg.position = position
        self.pub_single.publish(msg)
        
    def _publish_commands(
        self,
        positions: List[float],
    ) -> None:

        from meloha.robot_utils import convert_angle_to_position # for solving circular import
        positions: Union[List, float] = convert_angle_to_position(positions)
        self.node.get_logger().debug(f'Publishing {positions=}')

        msg = MultiSetPosition()
        msg.ids = self.motor_id
        msg.positions = positions
        self.pub_group.publish(msg)
    
    def _check_collision(self, positions: List[float]):
        
        # Check collision
        x, y, z = self.current_ee_position

        if (-0.18 <= x <= 0.18) and (-0.58 <= y <= 0) and (-0.10 <= z <= 0.10):
            self.node.get_logger().error(
                f"[충돌 위험] End-effector 위치가 허용 범위를 벗어났습니다: x={x:.2f}, y={y:.2f}, z={z:.2f}"
            )
            rclpy.shutdown()
        return True

    def get_node(self) -> MelohaRobotNode:
        return self.node
    

def calculate_ik_computation_time():

    node = create_meloha_global_node('meloha')

    left_manipulator = Manipulator(side="left", node=node)

    robot_startup(node)

    workspace_dir = os.path.expanduser("~/meloha_ws")
    file_path = os.path.join(workspace_dir, 'src/meloha/joint_data/displacement.csv')
    displacements = np.loadtxt(file_path, delimiter=',', skiprows=1)

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

    node = create_meloha_global_node('meloha')

    left_manipulator = Manipulator(side="left", node=node)

    robot_startup(node)

    collision_target = [180, -100, -400]
    left_manipulator.current_ee_position = collision_target
    left_joint_angles = left_manipulator.solve_ik(collision_target)
    left_manipulator.set_joint_positions(left_joint_angles)

if __name__ == "__main__":
    check_collsion_function()
 





