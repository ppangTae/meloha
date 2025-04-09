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
)

from meloha.real_env import (
    make_real_env
)

from meloha.robot_utils import (
    ImageRecorder,
)

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

def capture_one_episode(
    dt,
    max_timesteps,
    camera_names,
    dataset_dir,
    dataset_name,
    overwrite,
):
    print(f'Dataset name: {dataset_name}')

    node = create_meloha_global_node('meloha')

    env = make_real_env(
        node=node,
        setup_robots=False,
    )

    robot_startup(node)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # Data collection
    actual_dt_history = []
    time0 = time.time()
    DT = 1 / FPS
    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        env.get_images()
        t1 = time.time()
        actual_dt_history.append([t0, t1])
        time.sleep(max(0, DT - (time.time() - t0)))
    print(f'Avg fps: {max_timesteps / (time.time() - time0)}')

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 20:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 20, re-collecting... \n\n\n\n')
        return False

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
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(
            DT,
            max_timesteps,
            camera_names,
            dataset_dir,
            dataset_name,
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
        required=True,
    )
    parser.add_argument(
        '--episode_idx',
        action='store',
        type=int,
        help='Episode index.',
        default=None,
        required=False,
    )
    main(vars(parser.parse_args()))
