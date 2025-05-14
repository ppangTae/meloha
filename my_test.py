import numpy as np
import math

def convert_angle_to_position(rad):
    """
        Convert radian angle(s) to ROBOTIS MOTOR Command Value(s)
        -pi -> -501923, pi -> 501923
        Supports scalar, list, or numpy array input.
    """
    max_input = math.pi
    max_output = 501923

    rad = np.asarray(rad)
    pos = (rad / max_input) * max_output
    result = pos.astype(int)
    
    if result.ndim == 0:
        return int(result)  # scalar
    else:
        return result.tolist()  # list of ints

def convert_position_to_angle(pos):
    """
    Convert ROBOTIS MOTOR Command Value(s) to radian angle(s)
    -501923 -> -pi, 501923 -> pi
    Supports scalar, list, or numpy array input.
    Always returns float or list[float].
    """
    max_input = 501923
    max_output = math.pi

    pos = np.asarray(pos)
    angle = (pos / max_input) * max_output

    if angle.ndim == 0:
        return float(angle)  # scalar
    else:
        return angle.tolist()  # list of floats
    
pos = np.array([200000, 100000])
print(convert_position_to_angle(pos))
print(convert_angle_to_position([1.2512822, 0.6259112]))

