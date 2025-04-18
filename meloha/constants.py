# flake8: noqa

import os

### Task parameters
# # RealSense cameras image topic (realsense2_camera v4.54)
# COLOR_IMAGE_TOPIC_NAME = '{}/color/image_rect_raw'

# RealSense cameras image topic (realsense2_camera v4.55 and up)
COLOR_IMAGE_TOPIC_NAME = '/usb_{}/image_raw'

DATA_DIR = os.path.expanduser('~/meloha_data')

### ALOHA Fixed Constants
FPS = 30
DT = 1.0 / FPS

try:
    from rclpy.duration import Duration
    from rclpy.constants import S_TO_NS
    DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)
except ImportError:
    pass

LEFT_ARM_JOINT_NAMES = ['left_joint1', 'left_joint2', 'left_joint3']
RIGHT_ARM_JOINT_NAMES = ['right_joint1', 'right_joint2', 'right_joint3']
START_LEFT_ARM_POSE = [0, 0, 0]
START_RIGHT_ARM_POSE = [0, 0, 0]

### Real hardware task configurations
TASK_CONFIGS = {

    ### Template
    # 'aloha_template':{
    #     'dataset_dir': [
    #         DATA_DIR + '/aloha_template',
    #         DATA_DIR + '/aloha_template_subtask',
    #         DATA_DIR + '/aloha_template_other_subtask',
    #     ], # only the first entry in dataset_dir is used for eval
    #     'stats_dir': [
    #         DATA_DIR + '/aloha_template',
    #     ],
    #     'sample_weights': [6, 1, 1],
    #     'train_ratio': 0.99, # ratio of train data from the first dataset_dir
    #     'episode_len': 1500,
    #     'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    # },

    'meloha_box_picking' : {
        'dataset_dir': DATA_DIR + '/meloha_box_picking',
        'episode_len' : 300,
        'camera_names' : ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    }
}
