import os
import sys
import numpy as np
import unittest
from unittest.mock import patch

import rclpy
import launch
import launch.actions
import launch_testing.actions
import launch_testing.markers

from meloha.manipulator import Manipulator
from meloha.robot import (
    get_meloha_global_node, 
    robot_shutdown,
    create_meloha_global_node
)

class TestManipulatorClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.robot_node = create_meloha_global_node()


    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.left_manipulator = Manipulator(
            side="left",
            is_debug=False,
            is_sim=True,
            node=self.robot_node)
        
        self.right_manipulator = Manipulator(
            side="right",
            is_debug=False,
            is_sim=True,
            node=self.robot_node)

    def tearDown(self):
        pass

    @patch("rclpy.shutdown")
    def test_check_collision_zones(self, mock_shutdown):

        cases = [
            # (name,  position,  expect_shutdown)
            ("body_zone",  np.array([0.10,  0.30, 0.00]), True),
            ("base_zone",  np.array([0.20, -0.70, 0.00]), True),
            ("safe_zone",  np.array([0.30,  0.30, 0.00]), False),
        ]

        # (1) 시나리오별 실행
        for name, pos, should_shutdown in cases:
            with self.subTest(zone=name):
                mock_shutdown.reset_mock()
                self.left_manipulator.current_ee_position = pos
                result = self.left_manipulator._check_collision()

                if should_shutdown:
                    mock_shutdown.assert_called_once()
                else:
                    mock_shutdown.assert_not_called()
                    self.assertTrue(result)
