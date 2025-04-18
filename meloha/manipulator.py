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

from rclpy.node import Node
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from meloha.robot import (
    MelohaRobotNode,
    create_meloha_global_node,
    robot_startup,
    robot_shutdown
)
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from dynamixel_sdk_custom_interfaces.msg import SetPosition

class Manipulator:

    def __init__(
        self,
        side: str,
        is_debug: bool = False,
        robot_name: Optional[str] = None,
        node_name: str = 'meloha_robot_manipulation',
        node: Node = None,
    ):
        
        self.side = side
        self.is_debug = is_debug
        self.robot_name = robot_name
        self.node_name = node_name
        self.joint_states: list = [0.0, 0.0, 0.0] # ! 이거 None으로 바꿔주는 것이 좋긴함. 왜냐하면 노드 생성하면 알아서 받음.
        self.joint_commands = None
        self.js_mutex = Lock()
        
        if side == 'left':
            self.motor_id = [5,6,7]
        elif side == 'right':
            self.motor_id = [0,1,3]
        else:
            raise ValueError(f"side변수가 left, right가 아닙니다. {self.side}를 입력함.")

        self.dh_param: dict = self._load_dh_params()
        self.theta = np.zeros(3) # 관절 회전 각 -> 초기 위치가 모두 0도임.
        self.target_position = None
        setattr(self, f'current_position', None)

        R = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        self.T_base = np.eye(4)
        self.T_base[0:3, 0:3] = R
        self.T_base[0:3, 3] = np.array([-100, 0, 1000])

        if node is None:
            self.robot_node = create_meloha_global_node(node_name)
        else:
            self.robot_node = node

        cb_group_dxl_core = ReentrantCallbackGroup()
        cb_group_kinematics = ReentrantCallbackGroup()

        # Dynamixel의 위치제어를 위한 publisher
        self.pub_single = self.robot_node.create_publisher(
            msg_type=SetPosition,
            topic=f'/set_position',
            qos_profile=10,
            callback_group=cb_group_dxl_core
        )
        node.get_logger().info(f"Manipulator follower {side} joint commands publisher is created!")

        # Dynamixel에서 오는 joint state를 받기 위한 subscriber
        self.sub_joint_states = self.robot_node.create_subscription(
            msg_type=JointState,
            topic=f'/follwer_{self.side}/joint_states',
            callback=self._joint_state_cb,
            qos_profile=10,
            callback_group=cb_group_dxl_core,
        )
        node.get_logger().info(f"Manipulator follower {side} joint states subscriber is created!")

        # VIVE Tracker로부터 변위를 받는 subscriber
        # 변위를 받자마자 cb를 통해 역기구학을 풀어 변위만큼 이동하기 위한 joint 각도를 알아낸다.
        self.sub_tracker_displacement = self.robot_node.create_subscription(
            msg_type=PoseStamped,
            topic = f'/follwer_{self.side}/displacement',
            callback=self.sub_tracker_displacement_cb,
            qos_profile=10,
            callback_group=cb_group_kinematics
        )
        node.get_logger().info(f"Manipulator follower {side} displacement subscriber is created!")

        # Find initial starting position.
        self.robot_node.get_logger().debug(
            f'Trying to find joint states on topic "follwer_{side}/joint_states"...'
        )
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self.robot_node)
        self.robot_node.get_logger().debug('Found joint states. Continuing...')
        _, _, self.P, _ = self._solve_fk(self.joint_states)
        setattr(self, "current_position", self.P[:,3])
        node.get_logger().info(f"Maniputor {self.side} is located in {getattr(self, 'current_position')}")
        node.get_logger().info(f"Manipulator {side} is created well!")
        
    def sub_tracker_displacement_cb(self, msg):
        dx, dy, dz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        displacement = np.array([dx, dy, dz])
        if self.side == "left":
            self.target_position = self.left_current_position + displacement
        elif self.side == "right":
            self.target_position = self.right_current_position + displacement
        self.joint_commands = self._solve_ik(self.target_position)
        return
    
    def _joint_state_cb(self, msg: JointState):
        """
        Get the latest JointState message through a ROS Subscriber Callback.

        :param msg: JointState message
        """
        with self.js_mutex:
            self.joint_states = msg
    
    def get_node(self) -> MelohaRobotNode:
        return self.robot_node
    
    def dh_to_transform(self, alpha, a, d, theta):
        """DH 파라미터를 4x4 변환 행렬로 변환하는 함수."""
        ca = np.cos(np.deg2rad(alpha))
        sa = np.sin(np.deg2rad(alpha))
        ct = np.cos(theta)
        st = np.sin(theta)
        
        T = np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])
        return T

    def _load_dh_params(self):
        if self.side == "left":
            yaml_file_name = "left_dh_param.yaml"
        elif self.side == "right":
            yaml_file_name = "right_dh_param.yaml"
        else:
            raise ValueError(f"Unknown side: {self.side}")

        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        parent_dir = os.path.dirname(current_dir)
        yaml_file_path = os.path.join(parent_dir, 'config', yaml_file_name)
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        return data['dh_params']
    
    def _solve_ik(self, target):
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
            self.get_logger().error(f'IK 계산 실패: {str(e)}')
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

        return A, T, P, R

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
            msg.position = int((self.joint_commands[idx] / 360.0) * 600000)
            self.pub_single.publish(msg)

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
 





