#!/usr/bin/env python3

import argparse
import os
import time

from meloha.constants import (
    DT,
    FPS,
    JOINT_NAMES,
)
from meloha.real_env import (
    make_real_env,
)

import h5py
from meloha.robot import (
    create_meloha_global_node,
    robot_shutdown,
    robot_startup,
)


def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    node = create_meloha_global_node('meloha')

    env = make_real_env(node)

    robot_startup(node)

    env.reset()

    time0 = time.time()
    for action in actions:
        time1 = time.time()
        env.step(action)
        time.sleep(max(0, DT - (time.time() - time1)))
    print(f'Avg fps: {len(actions) / (time.time() - time0)}')

    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        action='store',
        type=str,
        help='Dataset dir.',
        required=True,
    )
    parser.add_argument(
        '--episode_idx',
        action='store',
        type=int,
        help='Episode index.',
        default=0,
        required=False,
    )
    main(vars(parser.parse_args()))
