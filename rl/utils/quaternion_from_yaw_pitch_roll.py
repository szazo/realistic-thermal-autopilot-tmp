from typing import Literal
import numpy as np
import numpy.typing as npt
import quaternion

from .vector import Vector3, VectorNx3

RotationType = Literal['extrinsic', 'intrinsic']


def quaternions_from_yaw_pitch_roll(
        yaw_pitch_roll_body_to_earth_rad: VectorNx3,
        type: RotationType) -> npt.NDArray[quaternion.quaternion]:

    q_array = np.array([
        quaternion_from_yaw_pitch_roll(yaw_pitch_roll_body_to_earth_rad[i, :],
                                       type=type)
        for i in range(yaw_pitch_roll_body_to_earth_rad.shape[0])
    ])

    return q_array


def quaternion_from_yaw_pitch_roll(
        yaw_pitch_roll_earth_to_body_rad: Vector3,
        type: RotationType) -> quaternion.quaternion:

    q_roll, q_pitch, q_yaw = _quaternion_components_from_yaw_pitch_roll(
        yaw_pitch_roll_earth_to_body_rad)

    if type == 'extrinsic':
        # if the input angles are extrinsic angles
        q = q_yaw * q_pitch * q_roll
    elif type == 'intrinsic':
        # if the input angles are intrinsic angles
        q = q_roll * q_pitch * q_yaw

    return q


def _quaternion_components_from_yaw_pitch_roll(
        yaw_pitch_roll_earth_to_body_rad: Vector3) -> quaternion.quaternion:

    roll = yaw_pitch_roll_earth_to_body_rad[2]
    pitch = yaw_pitch_roll_earth_to_body_rad[1]
    yaw = yaw_pitch_roll_earth_to_body_rad[0]

    q_roll = quaternion.from_rotation_vector(np.array([1., 0., 0.]) * roll)
    q_pitch = quaternion.from_rotation_vector(np.array([0., 1., 0.]) * pitch)
    q_yaw = quaternion.from_rotation_vector(np.array([0., 0., 1.]) * yaw)

    return q_roll, q_pitch, q_yaw
