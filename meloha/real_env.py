import collections
import time

from meloha.constants import (
    DT,
    START_LEFT_ARM_POSE,
    START_RIGHT_ARM_POSE,
)

from meloha.robot_utils import (
    ViveTracker,
    ImageRecorder,
    Recorder,
)

from meloha.manipulator import Manipulator

from meloha.robot import (
    create_meloha_global_node,
    get_meloha_global_node,
    MelohaRobotNode,
)

import matplotlib.pyplot as plt
import numpy as np

class RealEnv:
    
    """
    Environment for real robot bi-manual manipulation.

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
            "cam_high": (480x640x3),        # h, w, c, dtype='uint8'
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
            robot_name='follower_left',
            node=node,
        )
        self.follower_bot_right = Manipulator(
            side="right",
            robot_name='follower_right',
            node=node,
        )

        self.recorder_left = Recorder('left', node=node)
        self.recorder_right = Recorder('right', node=node)
        self.image_recorder = ImageRecorder(node=node)


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
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.follower_bot_left.set_joint_positions(left_action, blocking=False)
        self.follower_bot_right.set_joint_positions(right_action, blocking=False)
        if get_obs:
            obs = self.get_observation()
        else:
            obs = None
        return obs

def get_action(
    follower_bot_left: Manipulator,
    follower_bot_right: Manipulator,
):
    action = np.zeros(6)    
    action[:3] = follower_bot_left.arm_command
    action[3:] = follower_bot_right.arm_command    
    return action

def make_real_env(
    node: MelohaRobotNode = None,
):
    if node is None:
        node = get_meloha_global_node()
        if node is None:
            node = create_meloha_global_node('meloha')
    env = RealEnv(
        node=node,
    )
    node.get_logger().info("Environment class is initialized!")
    return env

if __name__ == '__main__':
    test_real_teleop()
