from dataclasses import dataclass
import numpy as np
from .api import (BirdTrajectoryMeta)


@dataclass
class InitialAndCutoffAltitudeParams:
    start_altitude_offset_relative_to_bird_min_m: float
    success_altitude_offset_relative_to_bird_max_m: float
    fail_altitude_offset_relative_to_start_m: float


@dataclass
class InitialAndCutoffAltitude:
    start_altitude_m: float
    success_altitude_m: float
    fail_altitude_m: float


def calculate_ai_initial_and_cutoff_altitude(
        params: InitialAndCutoffAltitudeParams, bird_meta: BirdTrajectoryMeta):

    start_altitude_m = bird_meta.minimum_altitude_m + params.start_altitude_offset_relative_to_bird_min_m
    success_altitude_m = bird_meta.maximum_altitude_m + params.success_altitude_offset_relative_to_bird_max_m
    fail_altitude_m = float(
        np.max((start_altitude_m +
                params.fail_altitude_offset_relative_to_start_m, 10.)))

    return InitialAndCutoffAltitude(start_altitude_m=start_altitude_m,
                                    success_altitude_m=success_altitude_m,
                                    fail_altitude_m=fail_altitude_m)
