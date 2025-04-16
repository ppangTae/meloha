import numpy as np
import os
import yaml

def load_dh_params(side: str):
    if side == "left":
        yaml_file_name = "left_dh_param.yaml"
    elif side == "right":
        yaml_file_name = "right_dh_param.yaml"
    else:
        raise ValueError(f"Unknown side: {side}")

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_dir = os.path.dirname(current_dir)
    yaml_file_path = os.path.join(parent_dir, 'config', yaml_file_name)
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data['dh_params']

dh_param = load_dh_params("left")
print(dh_param['joint1']['alpha'])
