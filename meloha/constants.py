# flake8: noqa

import os

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

LEFT_ARM_JOINT_NAMES = ['joint_1_left', 'joint_2_left', 'joint_3_left']
RIGHT_ARM_JOINT_NAMES = ['joint_1_right', 'joint_2_right', 'joint_3_right']
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3']
MOTOR_ID = {'left' : [5, 6, 7], 'right' : [0 ,1 ,3]}
LEFT_ARM_START_POSE = [0, -100000, 100000] # ROBOTIS MOTOR COMMAND. min:-501923(-pi), max: 501923(pi)
RIGHT_ARM_START_POSE = [0, 100000, -100000]

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
        'episode_len' : 900,
        'camera_names' : ['cam_head', 'cam_left_wrist','cam_right_wrist']
    },
}
