import collections
import time
import sys

from meloha.constants import (
    DT,
    LEFT_ARM_START_POSE,
    RIGHT_ARM_START_POSE,
)

from meloha.manipulator import Manipulator

from meloha.robot_utils import (
    ViveTracker,
    ImageRecorder,
    Recorder,
    convert_angle_to_position,
    convert_position_to_angle,
)

from meloha.robot import (
    create_meloha_global_node,
    get_meloha_global_node,
    MelohaRobotNode,
)

# import matplotlib.pyplot as plt
import numpy as np
import math

class RealEnv:
    
    """
    Environment for real robot bi-manual manipulation.

    # Action is calculated by displacement of Vive Trackers.
    Action space: [
        left_arm_qpos (3),             # absolute joint position
        right_arm_qpos (3),            # absolute joint position
    ]

    Observation space: {
        "qpos": Concat[
            left_arm_qpos (3),          # absolute joint position
            right_arm_qpos (3),         # absolute joint position
        ]
        "images": {
            "cam_head": (480x640x3)         # h, w, c, dtype='uint8'
            "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
            "cam_right_wrist": (480x640x3)  # h, w, c, dtype='uint8'
        }
    """

    def __init__(
        self,
        node: MelohaRobotNode,
    ):
        """
            Initialize the Real Robot Environment
        """
        
        self.follower_bot_left = Manipulator(
            side="left",
            node=node,
        )
        self.follower_bot_right = Manipulator(
            side="right",
            node=node,
        )

        self.recorder_left = Recorder('left', node=node)
        self.recorder_right = Recorder('right', node=node)
        self.image_recorder = ImageRecorder(node=node)

        self.max_allowed_joint_angle = np.deg2rad(10)

    def get_qpos(self):
        left_arm_qpos = self.recorder_left.qpos
        right_arm_qpos= self.recorder_right.qpos
        return np.concatenate(
            [left_arm_qpos, right_arm_qpos]
        )


    def get_images(self):
        return self.image_recorder.get_images()


    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['images'] = self.get_images()
        
        return obs


    def step(self, action, get_obs=True):


        current_left_positions = self.follower_bot_left.joint_states
        current_right_positions = self.follower_bot_right.joint_states
        target_left_positions = action[:3]
        target_right_positions = action[3:]

        delta_left = target_left_positions - current_left_positions
        delta_right = target_right_positions - current_right_positions

        self.follower_bot_left.set_joint_positions(action[:3])
        self.follower_bot_right.set_joint_positions(action[3:])
        if get_obs:
            obs = self.get_observation()
        else:
            obs = None
        return obs
    

    def reset(self):

        # TODO : 초기위치로 돌아가는 로직을 추가해야함.

        left_arm_joint_states = self.follower_bot_left.joint_states
        initial_joint_states: list = convert_position_to_angle(LEFT_ARM_START_POSE)
        for js, init_js in zip(left_arm_joint_states, initial_joint_states):
            # TODO : 오류를 발생시킬게 아니라, 원래 자리로 돌아가도록 해야함. 그래야 데이터수집하기가 편함.
            if not math.isclose(js, init_js, abs_tol=1e-3):  # 오차 허용 범위 ±0.001
                raise ValueError(f"left arm의 {initial_joint_states=}인데 {left_arm_joint_states=}에 있습니다.\n \
                                   초기위치에 정확히 도착하지 않았습니다.")
            
        right_arm_joint_states = self.follower_bot_right.joint_states
        initial_joint_states: list = convert_position_to_angle(RIGHT_ARM_START_POSE)
        for js, init_js in zip(right_arm_joint_states, initial_joint_states):
            if not math.isclose(js, init_js, abs_tol=1e-3):  # 오차 허용 범위 ±0.001
                raise ValueError(f"right arm의 {initial_joint_states=}인데 {right_arm_joint_states=}에 있습니다.\n \
                                   초기위치에 정확히 도착하지 않았습니다.")
            
        obs = self.get_observation()
        return obs


def get_action(
    tracker_left: ViveTracker,
    tracker_right: ViveTracker,
    follower_bot_left: Manipulator,
    follower_bot_right: Manipulator,
):
    moving_scale = 3.0
    left_displacement = moving_scale * tracker_left.displacement
    right_displacement = moving_scale * tracker_right.displacement

    left_displacement[1] = -left_displacement[1] # Y-axis inversion for left side
    right_displacement[1] = -right_displacement[1] # Y-axis inversion for left side

    left_ee_target = follower_bot_left.current_ee_position + left_displacement
    right_ee_target = follower_bot_right.current_ee_position + right_displacement

    left_ik_success, left_action = follower_bot_left.solve_ik(left_ee_target)
    right_ik_success, right_action = follower_bot_right.solve_ik(right_ee_target)

    if left_ik_success is False or right_ik_success is False:
        action = np.concatenate([
            follower_bot_left.joint_states,
            follower_bot_right.joint_states
        ])
    else:
        action = np.concatenate([left_action, right_action])

    return action


def make_real_env(
    node: MelohaRobotNode = None,
):
    if node is None:
        node = get_meloha_global_node()
        if node is None:
            node = create_meloha_global_node('meloha')
    env = RealEnv(node=node)
    node.get_logger().info("Environment class is initialized!")
    return env

if __name__ == '__main__':
    test_real_teleop()
