import numpy as np
from utils import Vector3


def calculate_resolution(box_size: Vector3 | float,
                         spacing_m: float | Vector3) -> Vector3 | int:
    result = (np.asarray(box_size, dtype=float) / spacing_m + 1).astype(int)
    if np.isscalar(box_size):
        return int(result)
    else:
        return result
