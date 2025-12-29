from dataclasses import dataclass
import logging
import numpy as np
import ray
from utils import Vector3, VectorNx3, VectorN
from .realistic_air_velocity_field_interface import DecomposedRealisticAirVelocityFieldInterface, RealisticAirVelocityFieldInterface, DecomposedThermalCoreSpline


@dataclass
class RealisticAirVelocityFieldRayParameters:
    actor_count: int = 3
    coordinate_per_actor: int = 50_000


@ray.remote
class RayActor:

    _air_velocity_field: RealisticAirVelocityFieldInterface

    def __init__(self, air_velocity_field: RealisticAirVelocityFieldInterface):
        self._air_velocity_field = air_velocity_field

    def get_velocity(
        self,
        X: VectorNx3,
        t: float = 0,
        include: str | list | np.ndarray | None = None,
        exclude: str | list | np.ndarray | None = None,
        relative_to_ground: bool = True,
        return_components: bool = False,
    ):
        return self._air_velocity_field.get_velocity(
            X=X,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=return_components)


# Ray parallelism for querying realistic air velocity field (for visualization)
class RealisticAirVelocityFieldRayAdapter(
        DecomposedRealisticAirVelocityFieldInterface):

    _wrapped_field: DecomposedRealisticAirVelocityFieldInterface | RealisticAirVelocityFieldInterface
    _params: RealisticAirVelocityFieldRayParameters

    _log: logging.Logger

    def __init__(self,
                 wrapped_field: DecomposedRealisticAirVelocityFieldInterface
                 | RealisticAirVelocityFieldInterface,
                 params: RealisticAirVelocityFieldRayParameters):

        self._wrapped_field = wrapped_field
        self._params = params

        self._log = logging.getLogger(__class__.__name__)

    def get_velocity(self,
                     X: Vector3 | VectorNx3,
                     t: float = 0,
                     include: str | list | np.ndarray | None = None,
                     exclude: str | list | np.ndarray | None = None,
                     relative_to_ground: bool = True,
                     return_components: bool = False):

        self._log.debug('get_velocity; X shape=%s,t=%s', X.shape, t)

        X = np.array(X)
        if X.ndim == 1 or X.shape[0] < self._params.coordinate_per_actor:
            # single coordinate or not too much coordinates, use without ray
            self._log.debug(
                'coordinate_count=%s < coordinate_per_actor=%s; skip ray',
                X.shape[0], self._params.coordinate_per_actor)

            return self._wrapped_field.get_velocity(
                X=X,
                t=t,
                include=include,
                exclude=exclude,
                relative_to_ground=relative_to_ground,
                return_components=return_components)

        # we have more coordinates, create actors
        coordinate_count = X.shape[0]
        package_count = int(
            np.ceil(coordinate_count / self._params.coordinate_per_actor))
        actor_count = np.min((package_count, self._params.actor_count))

        self._log.debug('coordinate_count=%s,package_count=%s,actor_count=%s',
                        coordinate_count, package_count, actor_count)

        actors = [
            RayActor.remote(self._wrapped_field) for _ in range(actor_count)
        ]

        futures = []
        for i in range(package_count):
            # spread among the actors
            package_size = self._params.coordinate_per_actor
            package_coords = X[i * package_size:i * package_size +
                               package_size, :]

            actor_index = i % actor_count
            actor_future = actors[actor_index].get_velocity.remote(
                X=package_coords,
                t=t,
                include=include,
                exclude=exclude,
                relative_to_ground=relative_to_ground,
                return_components=return_components)
            futures.append(actor_future)

        self._log.debug('futures length=%s', len(futures))

        # get the result from the actors
        results = ray.get(futures)

        if return_components:
            velocities = np.concatenate([item[0] for item in results])
            out_components = {}
            for item in results:
                component_dict = item[1]
                for key, value in component_dict.items():
                    if key not in out_components:
                        out_components[key] = value
                    else:
                        out_components[key] = np.concatenate(
                            (out_components[key], value))

            return velocities, out_components
        else:
            velocities = np.concatenate(results)

    def get_thermal_core(self, z: VectorN | float, t: float = 0, **kwargs):
        return self._wrapped_field.get_thermal_core(z=z, t=t, **kwargs)

    @property
    def current_thermal_core_spline(self) -> DecomposedThermalCoreSpline:

        assert self._wrapped_field.current_thermal_core_spline, "it is not a decomposed realistic air velocity field"

        return self._wrapped_field.current_thermal_core_spline
