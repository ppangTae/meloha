from typing import (
    Dict,
    List,
    Optional,
)

from rclpy.node import Node
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.logging import LoggingSeverity, set_logger_level
from meloha.robot import (
    create_meloha_global_node,
)

class Manipulator:

    def __init__(
        self,
        is_debug: bool = False,
        robot_name: Optional[str] = None,
        logging_level: LoggingSeverity = LoggingSeverity.INFO,
        node_name: str = 'meloha_robot_manipulation',
        node: Node = None,
    ):
        self.is_debug = is_debug
        self.robot_name = robot_name
        self.node_name = node_name

        if node is None:
            self.robot_node = create_meloha_global_node()
        else:
            self.robot_node = node
        
        set_logger_level(self.node_name, logging_level)

        self.robot_node.get_logger().debug((
            f"Created node with name= '{self.node_name}' in namespace='{robot_name}'"
        ))

        cb_group_dxl = ReentrantCallbackGroup()

        self.srv_set_op_modes = self.robot_node.create_client(
            srv_type=OperatingModes, # TODO : 서비스 통신을 위한 자신만의 srv 만들기
            srv_name=f'{self.ns}/set_operating_modes',
            callback_group=cb_group_dxl,
        )
        self.srv_set_pids = self.robot_node.create_client(
            srv_type=MotorGains,
            srv_name=f'{self.ns}/set_motor_pid_gains',
            callback_group=cb_group_dxl,
        )
        self.srv_set_reg = self.robot_node.create_client(
            srv_type=RegisterValues,
            srv_name=f'{self.ns}/set_motor_registers',
            callback_group=cb_group_dxl,
        )
        self.srv_get_reg = self.robot_node.create_client(
            srv_type=RegisterValues,
            srv_name=f'{self.ns}/get_motor_registers',
            callback_group=cb_group_dxl,
        )
        self.srv_get_info = self.robot_node.create_client(
            srv_type=RobotInfo,
            srv_name=f'{self.ns}/get_robot_info',
            callback_group=cb_group_dxl,
        )
        self.srv_torque = self.robot_node.create_client(
            srv_type=TorqueEnable,
            srv_name=f'{self.ns}/torque_enable',
            callback_group=cb_group_dxl,
        )
        self.srv_reboot = self.robot_node.create_client(
            srv_type=Reboot,
            srv_name=f'{self.ns}/reboot_motors',
            callback_group=cb_group_dxl,
        )

        # Check for xs_sdk by looking for set_operating_modes
        self.srv_set_op_modes.wait_for_service()
        self.srv_set_pids.wait_for_service()
        self.srv_set_reg.wait_for_service()
        self.srv_get_reg.wait_for_service()
        self.srv_get_info.wait_for_service()
        self.srv_torque.wait_for_service()
        self.srv_reboot.wait_for_service()





