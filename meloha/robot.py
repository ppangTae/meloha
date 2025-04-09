# Copyright 2024 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Contains a node and functions used to manage a generic robot's ROS node state."""

from threading import Thread
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

class MelohaRobotNode(Node):
    def __init__(
        self,
        node_name: str = 'meloha_robot_manipulation',
        namespace: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        self.node_name = node_name
        self.namespace = namespace
        super().__init__(node_name=self.node_name, namespace=self.namespace, *args, **kwargs)
        self.get_logger().info("Initialized MelohaRobotNode!")

    def wait_until_future_complete(self, future: Future, timeout_sec: float = None) -> None:
        """
        Block and wait for a future to complete.

        :param future: future to complete
        :param timeout_sec: timeout in seconds after which this will continue. if `None` or
            negative, will wait infinitely. defaults to `None`.
        """
        if meloha_is_up():
            rate = self.create_rate(20.0)
            if timeout_sec is None or timeout_sec < 0:
                while not future.done():
                    rate.sleep()
            else:
                start = self.get_clock().now()
                timeout_duration = Duration(seconds=timeout_sec)
                while not future.done() and self.get_clock().now() - start < timeout_duration:
                    rate.sleep()
        else:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

    def logdebug(self, message: str, **kwargs):
        self.get_logger().debug(message, **kwargs)

    def loginfo(self, message: str, **kwargs):
        self.get_logger().info(message, **kwargs)

    def logwarn(self, message: str, **kwargs):
        self.get_logger().warning(message, **kwargs)

    def logerror(self, message: str, **kwargs):
        self.get_logger().error(message, **kwargs)

    def logfatal(self, message: str, **kwargs):
        self.get_logger().fatal(message, **kwargs)


def __start(node: MelohaRobotNode, daemon: bool = True) -> None:
    """Start a background thread that spins the rclpy global executor."""
    global __meloha_is_up
    if meloha_is_up():
        raise Exception('Startup has already been requested.')
    __meloha_is_up = True
    global __meloha_execution_thread
    global __meloha_executor
    __meloha_executor = MultiThreadedExecutor()

    def spin(node: MelohaRobotNode) -> None:
        while rclpy.ok():
            __meloha_executor.add_node(node=node)
            __meloha_executor.spin()

    __meloha_execution_thread = Thread(target=spin, args=(node,), daemon=daemon)
    __meloha_execution_thread.start()


def robot_startup(
    node: Optional[MelohaRobotNode] = None,
    node_name: str = 'Meloha_robot_manipulation',
    namespace: Optional[str] = None,
) -> None:
    """
    Start up the Meloha Python API.

    :param node: (optional) The global node to start up the Meloha Python API with. If not
        given, this function will create it's own hidden global node. Defaults to `None`.
    :param node_name: (optional) The name of the node to create if no node was given. Defaults to
        `'Meloha_robot_manipulation'`.
    :param namespace: (optional) The namespace the node should be created under if no node was
        given. Defaults to `None`.
    """
    if node is None:
        # try to get global node
        node = get_meloha_global_node()
        # if no global node has been already created, create one here
        if node is None:
            node = create_meloha_global_node(node_name=node_name, namespace=namespace)
    __start(node)


def robot_shutdown(node: Optional[MelohaRobotNode] = None) -> None:
    """
    Destroy the node and shut down all threads and processes.

    :param node: (optional) The node to shutdown. If given `None`, will try to get and shut down
        the Meloha global node. Defaults to `None`.
    :note: If no node is given to this function, it is assumed that there is an Meloha global
        node.
    """
    if node is None:
        node = get_meloha_global_node()
    node.destroy_node()
    rclpy.shutdown()
    __meloha_execution_thread.join()
    if '__meloha_is_up' in globals():
        __meloha_is_up = False  # noqa: F841


def create_meloha_global_node(
    node_name: str = 'meloha_robot_manipulation',
    namespace: str = None,
    *args,
    **kwargs
) -> MelohaRobotNode:
    """
    Initialize the ROS context (if not already) and create a global generic node.

    :param node_name: The name of the node to create
    :param namespace: The namespace the node should be created under
    :return: A configured MelohaRobotNode
    """
    if '__meloha_global_node' in globals():
        raise Exception(
            'Tried to create an Meloha global node but one already exists.'
        )

    # Initialize the ROS context if not already
    if not rclpy.ok():
        rclpy.init(*args, **kwargs)

    # Instantiate a global InterbotixRobotNode
    global __meloha_global_node
    __meloha_global_node = MelohaRobotNode(
        node_name=node_name,
        namespace=namespace,
        *args,
        **kwargs,
    )
    return __meloha_global_node


def get_meloha_global_node() -> MelohaRobotNode:
    """Return the global Meloha node."""
    return __meloha_global_node


def meloha_is_up() -> bool:
    """
    Return the state of the Meloha Python-ROS API.

    :return: `True` if the API is up, `False` otherwise
    """
    if '__meloha_is_up' in globals():
        return __meloha_is_up
    else:
        return False
