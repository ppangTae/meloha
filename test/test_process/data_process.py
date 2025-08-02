# scripts/bringup_io.py
import rclpy, sys
from meloha.robot import create_meloha_global_node, robot_startup
from meloha.robot_utils import ImageRecorder, Recorder, ViveTracker

def main():
    rclpy.init()

    node = create_meloha_global_node('meloha')

    recorder_left = Recorder('left', node=node)
    recorder_right = Recorder('right', node=node)
    image_recorder = ImageRecorder(node=node)

    tracker_left = ViveTracker(
        side='left',
        tracker_sn='LHR-21700E73',
        node=node,
    )

    tracker_right = ViveTracker(
        side='right',
        tracker_sn='LHR-0B6AA285',
        node=node,
    )
    robot_startup(node)        # background spin‑thread
    rclpy.spin(node)           # block until ^C

if __name__ == '__main__':
    main()
