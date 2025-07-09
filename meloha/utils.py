import rclpy
from rclpy.logging import LoggingSeverity
import os
import time
import numpy as np
import cv2
import h5py
import pyfiglet

# 유효한 로그 레벨 매핑 딕셔너리
LOG_LEVEL_MAP = {
    'DEBUG': LoggingSeverity.DEBUG,
    'INFO': LoggingSeverity.INFO,
    'WARN': LoggingSeverity.WARN,
    'WARNING': LoggingSeverity.WARN,
    'ERROR': LoggingSeverity.ERROR,
    'FATAL': LoggingSeverity.FATAL,
    'CRITICAL': LoggingSeverity.FATAL,
}

def get_transformation_matrix(theta: float, alpha: float, a: float, d: float) -> np.ndarray:
    
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def normalize_log_level(level_str: str) -> int:
    """
    사용자 입력 문자열을 받아서 적절한 LoggingSeverity 상수로 변환

    :param level_str: 사용자 입력 (e.g., "debug", "INFO", "Warning")
    :return: rclpy.logging.LoggingSeverity 상수
    :raises: ValueError
    """
    level_str_upper = level_str.strip().upper()
    if level_str_upper not in LOG_LEVEL_MAP:
        raise ValueError(
            f"Invalid logging level: '{level_str}'. "
            f"Valid options are: {list(LOG_LEVEL_MAP.keys())}"
        )
    return LOG_LEVEL_MAP[level_str_upper]


def compress_hdf5(dataset_dir, dataset_name, max_timesteps=600,compress_quality=50):
    """
    Compress image data using JPEG and save to HDF5 format.

    Args:
        data_dict (dict): Dictionary containing keys like /observations/images/cam0, etc.
        dataset_path (str): Path prefix where .hdf5 will be saved (no extension).
        compress_quality (int): JPEG quality (0~100), lower is more compressed.
    """

    
    print("[*] load hdf5 file")
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    data_dict = {
        '/observations/qpos': [],
        '/action': [],
    }

    with h5py.File(dataset_path, 'r') as root:
        COMPRESS = root.attrs.get('compress', False) # TODO : compressed 되어있다면 에러 반환
        if COMPRESS:
            raise ValueError(f"Dataset '{dataset_name}' is already compressed.")
        COMPRESS = True
        print("Compression now enabled.")
        data_dict['/observations/qpos'] = root['/observations/qpos'][()]
        data_dict['/action'] = root['/action'][()]
        camera_names = list(root['/observations/images/'].keys())
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = root[f'/observations/images/{cam_name}'][()]

    # JPEG compression
    t0 = time.time()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_quality]  # tried as low as 20, seems fine
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
    compressed_path = os.path.join(dataset_dir, dataset_name + "_compressed")
    with h5py.File(compressed_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
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
        _ = obs.create_dataset('qpos', (max_timesteps, 6))
        _ = root.create_dataset('action', (max_timesteps, 6))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving: {time.time() - t0:.1f} secs')
    return True


def decode_and_save_images(hdf5_path, output_dir, save_to_disk=True, max_frames=None):
    """
    Decode compressed images from HDF5. Optionally save to disk or return as dict.

    Args:
        hdf5_path (str): Path to .hdf5 file.
        output_dir (str): Output directory to save images (if save_to_disk=True).
        save_to_disk (bool): If True, saves JPEGs to disk. If False, returns image dict.
        max_frames (int or None): Optional limit on number of frames to decode.

    Returns:
        image_dict (dict): If save_to_disk=False, returns dict of decoded images.
    """
    print("[*] Starting decompression...")
    os.makedirs(output_dir, exist_ok=True)
    image_dict = {}

    with h5py.File(hdf5_path, 'r') as root:
        camera_names = list(root['/observations/images'].keys())
        compress_len = root['/compress_len'][()]
        num_timesteps = root['/observations/images'][camera_names[0]].shape[0]

        for cam_idx, cam_name in enumerate(camera_names):
            cam_dir = os.path.join(output_dir, cam_name)
            if save_to_disk:
                os.makedirs(cam_dir, exist_ok=True)
            image_list = []

            for t in range(num_timesteps if max_frames is None else min(num_timesteps, max_frames)):
                padded_img = root[f'/observations/images/{cam_name}'][t]
                valid_len = compress_len[cam_idx][t]
                jpeg_data = padded_img[:valid_len]
                decoded = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    image_list.append(decoded)
                    if save_to_disk:
                        filename = os.path.join(cam_dir, f'{t:04d}.jpg')
                        cv2.imwrite(filename, decoded)
                else:
                    print(f"[!] Failed to decode timestep {t} for {cam_name}")

            image_dict[cam_name] = image_list

    print("[*] Finished decoding.")
    if not save_to_disk:
        return image_dict

def print_countdown(msg:str, start=5, delay=1):
    for i in range(start, 0, -1):
        # os.system('clear')  # macOS/Linux
        banner = pyfiglet.figlet_format(str(i))
        print(banner)
        time.sleep(delay)
    # os.system('clear')
    print(pyfiglet.figlet_format(msg))


