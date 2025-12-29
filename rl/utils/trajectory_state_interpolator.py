from typing import Literal
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import quaternion
from scipy.interpolate import interp1d

from .vector import Vector3, VectorN, VectorNx3
from .quaternion_from_yaw_pitch_roll import RotationType, quaternions_from_yaw_pitch_roll


@dataclass
class TrajectoryEulerInterpolatorInput:
    time: VectorN
    position_xyz: VectorNx3
    orientation_yaw_pitch_roll_rad: VectorNx3
    velocity_xyz: VectorNx3


@dataclass
class TrajectoryInterpolatorInput:
    time: VectorN
    position_xyz: VectorNx3
    orientation: npt.NDArray[quaternion.quaternion]
    velocity_xyz: VectorNx3


@dataclass
class State:
    position_xyz: Vector3
    orientation: quaternion.quaternion
    velocity_xyz: Vector3


def euler_interpolator_input_to_quaternion(
        input: TrajectoryEulerInterpolatorInput,
        rotation_type: RotationType) -> TrajectoryInterpolatorInput:
    quaternions = quaternions_from_yaw_pitch_roll(
        input.orientation_yaw_pitch_roll_rad, type=rotation_type)

    return TrajectoryInterpolatorInput(time=input.time,
                                       position_xyz=input.position_xyz,
                                       orientation=quaternions,
                                       velocity_xyz=input.velocity_xyz)


class TrajectoryStateInterpolator:

    _time: VectorN
    _position_velocity_interpolator: interp1d
    _orientation: npt.NDArray[quaternion.quaternion]

    def __init__(self, input: TrajectoryInterpolatorInput,
                 position_velocity_interpolation_type: Literal['linear',
                                                               'cubic']):

        self._time = input.time
        self._position_velocity_interpolator = self._create_position_velocity_interpolator(
            input, type=position_velocity_interpolation_type)
        self._orientation = input.orientation

    def query(self, time: float) -> State | None:
        position_velocity = self._position_velocity_interpolator(time)

        if np.all(np.isnan(position_velocity)):
            return None

        position = position_velocity[..., 0]
        velocity = position_velocity[..., 1]

        orientation = quaternion.squad(R_in=self._orientation,
                                       t_in=self._time,
                                       t_out=np.array(time),
                                       unflip_input_rotors=True)

        result = State(position_xyz=position,
                       orientation=orientation,
                       velocity_xyz=velocity)

        return result

    def _create_position_velocity_interpolator(
            self, trajectory: TrajectoryInterpolatorInput,
            type: Literal['linear', 'cubic']):

        data = np.stack((trajectory.position_xyz, trajectory.velocity_xyz),
                        axis=-1)

        interpolator = interp1d(x=trajectory.time, y=data, kind=type, axis=0)

        return interpolator
