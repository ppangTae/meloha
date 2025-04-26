#!/usr/bin/env python3

import argparse
import os
import time
import signal
from functools import partial

from meloha.constants import (
    DT,
    FPS,
    TASK_CONFIGS,
    START_RIGHT_ARM_POSE,
    START_LEFT_ARM_POSE,
)

from meloha.real_env import (
    make_real_env,
    get_action
)

from meloha.robot_utils import (
    ImageRecorder,
    ViveTracker,
    move_arms,
)

from meloha.manipulator import (
    Manipulator
)

from meloha.utils import normalize_log_level

import cv2
import h5py
from meloha.robot import (
    create_meloha_global_node,
    robot_shutdown,
    robot_startup,
)
import numpy as np
import rclpy
from tqdm import tqdm

def opening_ceremony(
    follower_bot_left: Manipulator,
    follower_bot_right: Manipulator,
):
    # move arms to starting position
    # TODO(준서) : 부딪히는 것을 방지하기 위한 팔 이동 방식을 생각해야함.
    # TODO        마지막 관절부터 0도로 만드는 것은 어떨련지?
    move_arms(
        [follower_bot_left, follower_bot_right],
        [START_LEFT_ARM_POSE, START_RIGHT_ARM_POSE],
        moving_time=1.5,
    )

    print('Started!')

def capture_one_episode(
    dt,
    max_timesteps,
    camera_names,
    dataset_dir,
    dataset_name,
    logging_level,
    overwrite,
):
    print(f'Dataset name: {dataset_name}')

    node = create_meloha_global_node('meloha')

    vive_tracker = ViveTracker(node=node)

    env = make_real_env(node=node)

    robot_startup(node)

    node.get_logger().set_level(logging_level)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    opening_ceremony(env.follower_bot_left, env.follower_bot_right)

    # Data collection
    node.get_logger().info("Data collection start\n")
    obs = env.reset()
    observations = [obs]
    actions = []
    actual_dt_history = []
    time0 = time.time()
    DT = 1 / FPS
    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = get_action(vive_tracker, env.follower_bot_left, env.follower_bot_right)
        t1 = time.time()
        obs = env.step(action)
        t2 = time.time()
        observations.append(obs)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))
    print(f'Avg fps: {max_timesteps / (time.time() - time0)}')

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 25:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 25, re-collecting... \n\n\n\n')
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'

    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/action': [],
    }

    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        obs = observations.pop(0)
        data_dict['/observations/qpos'].append(obs['qpos'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(
                obs['images'][cam_name]
            )

    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                # 0.02 sec # cv2.imdecode(encoded_image, 1)
                result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving: {time.time() - t0:.1f} secs')

    robot_shutdown()
    return True


def main(args: dict):
    # task_config = TASK_CONFIGS[args['meloha_box_picking']]
    task_config = TASK_CONFIGS['meloha_box_picking']
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    logging_level = args['logging_level']
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    print(f"logging_level : {logging_level}")
    while True:
        is_healthy = capture_one_episode(
            DT,
            max_timesteps,
            camera_names,
            dataset_dir,
            dataset_name,
            logging_level,
            overwrite,
        )
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')
        ):
            return i
    raise Exception(f'Error getting auto index, or more than {max_idx} episodes')


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    # dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print((
        f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} '
        f'Step env: {np.mean(step_env_time):.3f}')
    )
    return freq_mean


def debug():
    print('====== Debug mode ======')
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        image_recorder.print_diagnostics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_name',
        action='store',
        type=str,
        help='Task name.',
        default="meloha_box_picking",
        required=False,
    )
    parser.add_argument(
        '--episode_idx',
        action='store',
        type=int,
        help='Episode index.',
        default=None,
        required=False,
    )
    parser.add_argument(
        '--logging_level',
        type=normalize_log_level,  # 바로 변환!
        default='INFO',
        help='Logging level (DEBUG, INFO, WARN, ERROR, FATAL)',
    )
    main(vars(parser.parse_args()))
