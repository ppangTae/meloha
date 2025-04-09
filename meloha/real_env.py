import collections
import time

from meloha.constants import (
    DT,
    START_LEFT_ARM_POSE,
    START_RIGHT_ARM_POSE,
)

from meloha.robot_utils import (
    ImageRecorder,
    move_arms,
    Recorder,
    setup_follower_bot,
    setup_leader_tracker,
)

from meloha.manipulator import Manipulator

import dm_env
from meloha.robot import (
    create_meloha_global_node,
    get_meloha_global_node,
    MelohaRobotNode,
)

from vive_tracker.vive_tracker_node import ViveTrackerNode

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
        setup_robots: bool = True,
    ):
        """Initialize the Real Robot Environment

        :param node: The MelohaRobotNode to build the Meloha API on
        :param setup_robots: True to run through the arm setup process on init, defaults to True
        """
        self.follower_bot_left = Manipulator(
            robot_name='follower_left',
            node=node,
        )
        self.follower_bot_right = Manipulator(
            robot_name='follower_right',
            node=node,
        )

        self.recorder_left = Recorder('left', node=node)
        self.recorder_right = Recorder('right', node=node)
        self.image_recorder = ImageRecorder(node=node)

        if setup_robots:
            self.setup_robots()

    def setup_robots(self):
        setup_follower_bot(self.follower_bot_left)
        setup_follower_bot(self.follower_bot_right)

    def get_qpos(self):
        left_arm_qpos = self.recorder_left.qpos
        right_arm_qpos= self.recorder_right.qpos
        return np.concatenate(
            [left_arm_qpos, right_arm_qpos]
        )

    def get_images(self):
        return self.image_recorder.get_images()

    def _reset_joints(self):
        left_reset_position = START_LEFT_ARM_POSE
        right_reset_position = START_RIGHT_ARM_POSE
        move_arms(
            [self.bot_left, self.bot_right],
            [left_reset_position, right_reset_position],
            moving_time=1.0,
        )

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot follower robot gripper motors
            self.follower_bot_left.robot_reboot_motors('single', 'gripper', True)
            self.follower_bot_right.robot_reboot_motors('single', 'gripper', True)
            self._reset_joints()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action, base_action=None, get_base_vel=False, get_obs=True):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.follower_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.follower_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)

        if get_obs:
            obs = self.get_observation(get_base_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)


def get_action(
    leader_tracker_left: ViveTrackerNode,
    leader_tracker_right: ViveTrackerNode
):
    action = np.zeros(6)  # 3 joint, 2 arm
    # Arm actions
    action[:3] = leader_tracker_left.core.joint_states.position[:3]
    action[3:] = leader_tracker_right.core.joint_states.position[:3]

    return action


def make_real_env(
    node: MelohaRobotNode = None,
    setup_robots: bool = True,
):
    if node is None:
        node = get_meloha_global_node()
        if node is None:
            node = create_meloha_global_node('meloha')
    env = RealEnv(
        node=node,
        setup_robots=setup_robots,
    )
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.

    It first reads joint poses from both leader arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleop and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """
    onscreen_render = True
    render_cam = 'cam_left_wrist'

    node = get_meloha_global_node()

    # source of data
    leader_tracker_left = ViveTrackerNode(
        robot_model='wx250s',
        robot_name='leader_left',
        node=node,
    )
    leader_tracker_right = ViveTrackerNode(
        robot_model='wx250s',
        robot_name='leader_right',
        node=node,
    )
    setup_leader_tracker(leader_tracker_left)
    setup_leader_tracker(leader_tracker_right)

    # environment setup
    env = make_real_env(node=node)
    ts = env.reset(fake=True)
    episode = [ts]
    # visualization setup
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for _ in range(1000):
        action = get_action(leader_tracker_left, leader_tracker_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()
