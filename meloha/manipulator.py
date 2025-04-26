from typing import (
    Dict,
    List,
    Optional,
)

import os
import yaml
import numpy as np
import time
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

from meloha.constants import MOTOR_ID, START_ARM_POSE

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from dynamixel_sdk_custom_interfaces.msg import SetPosition

class Manipulator:

    def __init__(
        self,
        side: str,
        is_debug: bool = False,
        robot_name: Optional[str] = None,
        node: Node = None,
    ):

        self.side = side
        self.is_debug = is_debug
        self.robot_name = robot_name
        self.motor_id = MOTOR_ID[self.side]
        self.dh_param: dict = self._load_dh_params(self.side)

        self.joint_states: list = None
        self.initial_states: list = START_ARM_POSE

        self.target_ee_position = None
        self.current_ee_position = None
        self.displacement = None

        self.js_mutex = Lock()

        # ! 이런 코드는 없애야됨.(깨끗하게)
        R = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        self.T_base = np.eye(4)
        self.T_base[0:3, 0:3] = R
        self.T_base[0:3, 3] = np.array([-100, 0, 1000])

        if node is None:
            self.robot_node = create_meloha_global_node(node_name)
        else:
            self.robot_node = node

        manipulator_cb_group = ReentrantCallbackGroup()

        # Dynamixel의 위치제어를 위한 publisher
        self.pub_single = self.robot_node.create_publisher(
            msg_type=SetPosition, # ! JointGroupCommand로 변경
            topic=f'/set_position',
            qos_profile=10,
            callback_group=manipulator_cb_group
        )
        node.get_logger().info(f"Manipulator follower {side} joint commands publisher is created!")

        # Dynamixel에서 오는 joint state를 받기 위한 subscriber
        self.sub_joint_states = self.robot_node.create_subscription(
            msg_type=JointState,
            topic=f'/follwer_{self.side}/joint_states',
            callback=self._joint_state_cb,
            qos_profile=10,
            callback_group=manipulator_cb_group,
        )
        node.get_logger().info(f"Manipulator follower {side} joint states subscriber is created!")

        # VIVE Tracker로부터 변위를 받는 subscriber
        self.sub_tracker_displacement = self.robot_node.create_subscription(
            msg_type=PoseStamped,
            topic = f'/follwer_{self.side}/displacement',
            callback=self._tracker_disp_cb,
            qos_profile=10,
            callback_group=manipulator_cb_group
        )
        node.get_logger().info(f"Manipulator follower {side} displacement subscriber is created!")

        # Find current joint_positions and ee position
        self.robot_node.get_logger().debug(
            f'Trying to find joint states on topic "follwer_{side}/joint_states"...'
        )
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self.robot_node)
        self.robot_node.get_logger().debug('Found joint states. Continuing...')
        self.P = self._solve_fk(self.joint_states)
        self.current_ee_position = self.P[:,3]
        node.get_logger().info(f"Maniputor {self.side} is located in {self.current_ee_position}")
        node.get_logger().info(f"Manipulator {side} is created well!")
        
    def _tracker_disp_cb(self, msg: PoseStamped):
        """
        Get the latest Vive Tracker displacement message through a ROS Subscriber Callback.

        :param msg: PosStamped message
        """
        dx, dy, dz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        self.displacement = np.array([dx, dy, dz])
        return
    
    def _joint_state_cb(self, msg: JointState):
        """
        Get the latest JointState message through a ROS Subscriber Callback.

        :param msg: JointState message
        """
        with self.js_mutex:
            self.joint_states = msg

    def _load_dh_params(self, side):
        yaml_file_name = f"{side}_dh_param.yaml"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        yaml_file_path = os.path.join(parent_dir, 'config', yaml_file_name)
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data['dh_params']
    
    def solve_ik(self, target):

        # TODO(준서)
        # 로봇의 안전문제를 잘 고려해야함.
        # 1. 역기구학을 풀면서 발생하는 수치적 오류 -> NaN발생
        # 2. workspace밖에 도달했을 때
        # 3. 로봇이 몸체에 부딪히지 않도록 특정영역에 들어오면 오류를 발생시켜 부딪히지 않도록 하기

        try:
            P2 = self.P[:,1] # base 좌표계에서 바라본 2번째 joint의 위치
            l2 = self.dh_param['joint2']['a']
            l3 = self.dh_param['joint3']['a']
            
            #목표 지점까지의 거리
            dist = np.linalg.norm(target - P2)
            # YZ 평면에 투영된 거리 계산
            proj = np.linalg.norm([target[1]-P2[1], target[2]-P2[2]])

            # theta_1 (joint1)
            denom = np.linalg.norm([P2[2]-target[2], P2[1]-target[1]])
            if denom < 1e-6: # 분모가 0에 가까울 때
                theta1 = 0.0
            else:
                theta1 = -np.arccos((P2[2] - target[2]) / denom)
                if target[1] < P2[1]:
                    theta1 = -theta1

            # theta_2 (joint2)
            alpha = np.arccos((l3**2 - l2**2 - dist**2) / (-2*l2*dist))
            beta = np.arccos(proj / dist)
            theta2 = beta + alpha if target[0] >= P2[0] else -(beta - alpha)

            # theta_3 (joint3)
            gamma = np.arccos((dist**2 - l2**2 - l3**2) / (-2*l2*l3))
            theta3 = -(np.pi - gamma)

            # 계산된 관절 각도 반환 (4번째 관절은 0으로 설정)
            return np.array([theta1, theta2, theta3])
        except Exception as e:
            self.robot_node.get_logger().error(f'IK 계산 실패: {str(e)}')
            return np.array([np.nan, np.nan, np.nan])
        
    
    def _solve_fk(self, positions: np.ndarray):
        # 순방향 기구학 계산
        i = 3 # joint 수
        j = i + 1
        
        #변환 행렬 초기화
        A = np.zeros((4, 4, i))
        T = np.zeros((4, 4, j))
        P = np.zeros((3, j))
        R = np.zeros((3, 3, j))

        # 베이스 변환 행렬 설정
        T[:, :, 0] = self.T_base
        P[:, 0] = self.T_base[0:3, 3]

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

    def set_joint_positions(
        self,
        joint_positions: List[float],
    ) -> bool:

        self.robot_node.get_logger().debug(f'Setting {joint_positions=}')
        self._publish_commands(joint_positions)
        
    def _publish_commands(
        self,
        positions: List[float],
    ) -> None:

        self.robot_node.get_logger().debug(f'Publishing {positions=}')
        self.joint_commands = list(positions)
        for idx in range(3):
            msg = SetPosition()
            msg.id = self.motor_id[idx]
            msg.position = int((self.joint_commands[idx] / 360.0) * 600000) # TODO : 이거맞나? 확인좀
            self.pub_single.publish(msg)

    def get_node(self) -> MelohaRobotNode:
        return self.robot_node

def calculate_ik_computation_time():

    node = create_meloha_global_node('meloha')

    manipulator = Manipulator(side="left", node=node)

    robot_startup(node)

    workspace_dir = os.path.expanduser("~/meloha_ws")
    file_path = os.path.join(workspace_dir, 'src/meloha/joint_data/displacement.csv')
    displacements = np.loadtxt(file_path, delimiter=',', skiprows=1)

    left_disp = displacements[1:, :3]  # 160x3
    durations = []

    # 초기 위치 설정
    current_pos = getattr(manipulator, "current_position")

    for disp in left_disp:
        target_position = current_pos + disp
        
        start = time.perf_counter()
        joint_angles = manipulator._solve_ik(target_position)
        end = time.perf_counter()

        manipulator.joint_states = joint_angles
        setattr(manipulator, "current_position", target_position)

        durations.append(end - start)

    avg_time_ms = np.mean(durations) * 1000
    print(f"총 {len(left_disp)}개의 IK 평균 계산 시간: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    calculate_ik_computation_time()
 





