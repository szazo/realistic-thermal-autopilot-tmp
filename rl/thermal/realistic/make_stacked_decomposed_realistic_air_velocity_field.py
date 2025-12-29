import os
import logging
from .load_decomposed_extrapolated_air_velocity_field import (
    load_decomposed_extrapolated_air_velocity_field)
from .stacked_decomposed_realistic_air_velocity_field import StackedDecomposedRealisticAirVelocityField
from .realistic_air_velocity_field_adapter import RealisticAirVelocityFieldAdapter
from .realistic_air_velocity_field_ray_adapter import (
    RealisticAirVelocityFieldRayAdapter)
from .stacked_decomposed_realistic_gaussian_filter_adapter import (
    StackedDecomposedRealisticGaussianFilterAdapter)
from .stacked_decomposed_realistic_air_velocity_field_params import (
    StackedDecomposedRealisticAirVelocityFieldParameters)


def make_stacked_decomposed_realistic_air_velocity_field(
    params: StackedDecomposedRealisticAirVelocityFieldParameters
) -> RealisticAirVelocityFieldAdapter:

    log = logging.getLogger(__name__)

    decomposition_path = os.path.abspath(params.path)
    log.debug(
        'making stacked decomposed realistic air velocity field using path "%s"...',
        decomposition_path)

    if not os.path.exists(decomposition_path):
        raise Exception(
            f'Decomposition path "{decomposition_path}" does not exist.')

    assert params.max_extrapolated_distance is not None, 'missing required "max_extrapolated_distance" parameter'

    wrapped_field = load_decomposed_extrapolated_air_velocity_field(
        path=decomposition_path,
        max_extrapolated_distance=params.max_extrapolated_distance)

    if params.enable_ray_parallelism:
        assert params.ray_parallelism_params is not None, 'ray_prallelism_params is missing'

        wrapped_field = RealisticAirVelocityFieldRayAdapter(
            wrapped_field=wrapped_field, params=params.ray_parallelism_params)

    stacked_field = StackedDecomposedRealisticAirVelocityField(wrapped_field)

    if params.stack_gaussian_filter_params is not None:
        stacked_field = StackedDecomposedRealisticGaussianFilterAdapter(
            wrapped_field=stacked_field,
            params=params.stack_gaussian_filter_params)

    air_velocity_field = RealisticAirVelocityFieldAdapter(
        stacked_field, params.name)
    log.debug('air velocity field created')

    return air_velocity_field
