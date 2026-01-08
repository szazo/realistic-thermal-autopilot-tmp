from typing import Literal
import numpy as np
import logging
from .realistic_air_velocity_field_interface import (
    RealisticAirVelocityFieldInterface,
    DecomposedRealisticAirVelocityFieldInterface)
from object.air import ReconstructedAirVelocityField


class StackedDecomposedRealisticAirVelocityField(
        RealisticAirVelocityFieldInterface):

    _field: DecomposedRealisticAirVelocityFieldInterface

    # the minimum z of the original thermal
    _z_min: float

    # the maximum z of the original thermal
    _z_max: float

    # the divisor used to determine the modulus (z region size)
    _z_divisor: float

    # xy difference between the bottom and top of the core
    _core_xy_diff: np.ndarray[Literal[2], float]

    def __init__(self, realistic_air_velocity_field:
                 DecomposedRealisticAirVelocityFieldInterface):

        self._log = logging.getLogger(__class__.__name__)

        self._field = realistic_air_velocity_field
        self._initialize_region()

    def _initialize_region(self):
        self._z_min = self._field.current_thermal_core_spline["X"].x_min
        self._z_max = self._field.current_thermal_core_spline["X"].x_max
        self._log.debug("z_min=%s, z_max=%s", self._z_min, self._z_max)

        # z_size_decimal = Decimal(str(self._z_max - self._z_min))
        # z_size_decimal_places = abs(z_size_decimal.as_tuple().exponent)
        # epsilon = 1.0 / (10.0**z_size_decimal_places)

        # self._z_divisor = self._z_max - self._z_min + epsilon
        self._z_divisor = self._z_max - self._z_min
        self._log.debug("z_divisor=%s", self._z_divisor)

        # t=0 because decomposed thermal is static
        core_xy_at_z_min = self._field.get_thermal_core(z=self._z_min, t=0)
        core_xy_at_z_max = self._field.get_thermal_core(z=self._z_max, t=0)
        self._core_xy_diff = core_xy_at_z_max - core_xy_at_z_min

        self._log.debug(
            "core_xy_at_z_min=%s, core_xy_at_z_max=%s, core_xy_diff=%s",
            core_xy_at_z_min,
            core_xy_at_z_max,
            self._core_xy_diff,
        )

    @property
    def segment_size(self) -> float:
        return self._z_max - self._z_min

    def segment_relative_altitude(
            self, z: float | np.ndarray[float]) -> float | np.ndarray[float]:

        return (z - self._z_min) % self._z_divisor

    def get_velocity(
        self,
        X: np.ndarray[Literal[3], float] | np.ndarray[Literal["N", 3], float],
        t: float = 0,
        include: str | list | np.ndarray = None,
        exclude: str | list | np.ndarray = None,
        relative_to_ground: bool = True,
        return_components: bool = False,
    ):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape((1, 3))

        z = X[:, 2]

        # calculate the altitude in the original thermal
        mod = self.segment_relative_altitude(z) + self._z_min

        # calculate the stacked segment index (-X at bottom, +X at top of the original)
        segment_index = (z - self._z_min) // self._z_divisor
        segment_index = np.column_stack((segment_index, segment_index))

        # shift the core based on the segment index
        xy_shifted = X[:, 0:2] - (segment_index * self._core_xy_diff)

        # construct the coordinates in the original thermal's coordinate space
        xyz_shifted = np.column_stack((xy_shifted, mod))

        velocity, components = self._field.get_velocity(
            X=xyz_shifted,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=True,
        )

        # replace nans to zero in the components and add them up
        components = {
            k: np.nan_to_num(v, nan=0.0)
            for k, v in components.items()
        }
        velocity = ReconstructedAirVelocityField.add_velocities(
            np.array(list(components.values())).T).T

        if return_components:
            return velocity, components
        else:
            return velocity

    def get_thermal_core(self,
                         z: np.ndarray[float] | float,
                         t: float = 0,
                         **kwargs):

        mod = self.segment_relative_altitude(z) + self._z_min

        # calculate the segment index for each z,
        # 0 means the original segment, -x means x times below, +x means x times above
        segment_index = (z - self._z_min) // self._z_divisor
        segment_index = np.column_stack((segment_index, segment_index))

        core_xy = self._field.get_thermal_core(mod, t, **kwargs)
        resulting_core_xy = core_xy + (segment_index * self._core_xy_diff)

        return resulting_core_xy

    def info(self):
        return dict(original_thermal_z_min=self._z_min,
                    original_thermal_z_max=self._z_max)
