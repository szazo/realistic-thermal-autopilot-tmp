import warnings
import numpy as np
import pandas as pd

from object.air import ReconstructedAirVelocityField

from utils import Vector3, VectorNx3, VectorN
from .realistic_air_velocity_field_interface import (
    DecomposedRealisticAirVelocityFieldInterface, DecomposedThermalCoreSpline)


class ReconstructedAirVelocityFieldAdapter(
        DecomposedRealisticAirVelocityFieldInterface):

    _wrapped_field: ReconstructedAirVelocityField

    def __init__(self, field: ReconstructedAirVelocityField):
        self._wrapped_field = field

    def get_velocity(self,
                     X: Vector3 | VectorNx3,
                     t: float = 0,
                     include: str | list | np.ndarray | None = None,
                     exclude: str | list | np.ndarray | None = None,
                     relative_to_ground: bool = True,
                     return_components: bool = False):

        result = self._wrapped_field.get_velocity(
            X=X,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=return_components)

        return result

    def get_thermal_core(self, z: VectorN | float, t: float = 0, **kwargs):
        return self._wrapped_field.get_thermal_core(z=z, t=t, **kwargs)

    @property
    def current_thermal_core_spline(self) -> DecomposedThermalCoreSpline:
        return self._wrapped_field.current_thermal_core_spline


# based on thermalmodelling/src/scripts/visualization/air_velocity_field_extrapolation.py
def load_decomposed_extrapolated_air_velocity_field(
    path: str, max_extrapolated_distance: float
) -> DecomposedRealisticAirVelocityFieldInterface:

    # ignore deprecation warnings temporaly
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # set chained_assignment mode
    pd_chained_assignment_save = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    try:
        air_velocity_field = ReconstructedAirVelocityField.from_path(
            path_to_files=path,
            max_extrapolated_distance=max_extrapolated_distance,
        )

        adapter = ReconstructedAirVelocityFieldAdapter(air_velocity_field)
        return adapter

    finally:
        # reset chained_assignment mode
        pd.options.mode.chained_assignment = pd_chained_assignment_save
        warnings.resetwarnings()
