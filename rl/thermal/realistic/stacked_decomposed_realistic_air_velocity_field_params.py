from dataclasses import dataclass, field
from .realistic_air_velocity_field_ray_adapter import (
    RealisticAirVelocityFieldRayParameters)
from .stacked_decomposed_realistic_gaussian_filter_adapter import (
    StackedDecomposedRealisticGaussianFilterAdapterParameters)


@dataclass
class StackedDecomposedRealisticAirVelocityFieldParameters:
    name: str
    path: str
    enable_ray_parallelism: bool = False
    ray_parallelism_params: RealisticAirVelocityFieldRayParameters = field(
        default_factory=RealisticAirVelocityFieldRayParameters)
    stack_gaussian_filter_params: StackedDecomposedRealisticGaussianFilterAdapterParameters | None = None
    max_extrapolated_distance: float | None = None
