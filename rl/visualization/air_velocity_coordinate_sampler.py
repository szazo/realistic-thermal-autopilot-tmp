from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from typing import Any
import numpy as np

from env.glider.base import SimulationBoxParameters
from utils import VectorN
from thermal.api import AirVelocityFieldInterface
from distributions import Distribution, DistributionConfigBase
from utils.vector import Vector3
from .calculate_resolution import calculate_resolution
from .create_uniform_polar_grid import create_uniform_polar_grid
from .calculate_resolution import calculate_resolution


@dataclass(kw_only=True)
class AirVelocitySamplerConfigBase():
    _target_: str = 'missing'


class AirVelocityCoordinateSampler(ABC):

    @abstractmethod
    def sample(
        self, air_velocity_field: AirVelocityFieldInterface
    ) -> tuple[VectorN, VectorN, VectorN]:
        pass


@dataclass
class GridAirVelocitySamplerParameters:
    bounds: SimulationBoxParameters
    spacing_m: Any  # float | Vector3


@dataclass(kw_only=True)
class GridAirVelocitySamplerConfig(GridAirVelocitySamplerParameters,
                                   AirVelocitySamplerConfigBase):
    _target_: str = 'visualization.air_velocity_coordinate_sampler.GridAirVelocityCoordinateSampler'


class GridAirVelocityCoordinateSampler(AirVelocityCoordinateSampler):

    _log: logging.Logger

    _bounds: SimulationBoxParameters
    _spacing_m: float | Vector3

    def __init__(self, bounds: SimulationBoxParameters,
                 spacing_m: float | Vector3):
        self._log = logging.getLogger(__class__.__name__)

        self._bounds = bounds
        self._spacing_m = spacing_m

    def sample(
        self, air_velocity_field: AirVelocityFieldInterface
    ) -> tuple[VectorN, VectorN, VectorN]:

        bounds = self._bounds
        assert bounds.box_size is not None
        assert bounds.limit_earth_xyz_low_m is not None and bounds.limit_earth_xyz_high_m is not None

        spacing_m = self._spacing_m
        spacing_m_arr = np.asarray(spacing_m)
        assert np.isscalar(spacing_m) or spacing_m_arr.size == 3

        resolution = calculate_resolution(np.array(bounds.box_size),
                                          spacing_m=spacing_m_arr)
        assert isinstance(resolution, np.ndarray)
        assert resolution.size == 3

        self._log.debug('resolution=%s', resolution)

        # create the grid coordinates
        X, Y, Z = np.meshgrid(
            np.linspace(bounds.limit_earth_xyz_low_m[0],
                        bounds.limit_earth_xyz_high_m[0],
                        num=resolution[0]),
            np.linspace(bounds.limit_earth_xyz_low_m[1],
                        bounds.limit_earth_xyz_high_m[1],
                        num=resolution[1]),
            np.linspace(bounds.limit_earth_xyz_low_m[2],
                        bounds.limit_earth_xyz_high_m[2],
                        num=resolution[2]))
        x = X.ravel()
        y = Y.ravel()
        z = Z.ravel()

        return x, y, z


@dataclass
class PolarAirVelocitySamplerNoiseConfig:
    x: DistributionConfigBase
    y: DistributionConfigBase
    z: DistributionConfigBase


@dataclass
class PolarAirVelocitySamplerBaseParameters:
    r_max_m: float
    r_step_m: float
    z_min_m: float
    z_max_m: float
    z_step_m: float


@dataclass
class PolarAirVelocitySamplerNoise:
    x: Distribution
    y: Distribution
    z: Distribution


@dataclass(kw_only=True)
class PolarAirVelocitySamplerConfig(PolarAirVelocitySamplerBaseParameters,
                                    AirVelocitySamplerConfigBase):
    _target_: str = 'visualization.air_velocity_coordinate_sampler.PolarAirVelocityCoordinateSampler'
    noise: PolarAirVelocitySamplerNoiseConfig


class PolarAirVelocityCoordinateSampler(AirVelocityCoordinateSampler):

    _params: PolarAirVelocitySamplerBaseParameters
    _noise: PolarAirVelocitySamplerNoise

    def __init__(self, noise: PolarAirVelocitySamplerNoise, **base_params):
        self._params = PolarAirVelocitySamplerBaseParameters(**base_params)
        self._noise = noise

    def sample(
        self, air_velocity_field: AirVelocityFieldInterface
    ) -> tuple[VectorN, VectorN, VectorN]:

        params = self._params

        resolution = calculate_resolution(params.z_max_m - params.z_min_m,
                                          params.z_step_m)
        assert isinstance(resolution, int)

        z = np.linspace(params.z_min_m, params.z_max_m, num=resolution)

        # query core coordinates
        core_xy = air_velocity_field.get_thermal_core(z_earth_m=z, t_s=0)
        assert core_xy.shape == (len(z), 2)

        # create polar coordinates on a plane
        xs, ys = create_uniform_polar_grid(r_max_m=params.r_max_m,
                                           r_step_m=params.r_step_m)

        # repeat the x and y coordinates along the z axis
        xs = np.expand_dims(xs, axis=1)
        xs = np.repeat(xs, repeats=len(z), axis=1)
        ys = np.expand_dims(ys, axis=1)
        ys = np.repeat(ys, repeats=len(z), axis=1)

        # add the core coordinates along the z (planes will follow the core)
        xs = xs + core_xy[:, 0]
        ys = ys + core_xy[:, 1]

        # repeat z items for each plane
        zs = np.repeat(z, len(xs))

        # flatten the x and y coordinates
        xs = np.ravel(xs, order='F')
        ys = np.ravel(ys, order='F')

        # add random noise to the coordinates to help the volume interpolation
        noise = self._noise
        x_distribution = noise.x
        y_distribution = noise.y
        z_distribution = noise.z

        x_noise = x_distribution.sample(size=xs.size)
        y_noise = y_distribution.sample(size=ys.size)
        z_noise = z_distribution.sample(size=zs.size)

        xs += x_noise
        ys += y_noise
        zs += z_noise

        return xs, ys, zs
