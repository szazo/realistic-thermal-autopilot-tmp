from typing import Literal
from dataclasses import dataclass
import time
import numpy as np
import logging
from .realistic_air_velocity_field_interface import (
    RealisticAirVelocityFieldInterface,
    StackedRealisticAirVelocityFieldInterface)


@dataclass
class StackedDecomposedRealisticGaussianFilterAdapterParameters:
    filter_size_xyz_m: tuple[float, float, float]
    filter_spacing_xyz_m: tuple[float, float, float]
    filter_sigma_xyz_m: tuple[float, float, float]
    maximum_distance_from_core: float
    segment_bottom_altitude_region_m: float
    segment_top_altitude_region_m: float
    amplitude_multiplier: float
    zero_mask_enabled: bool = False
    zero_epsilon: float = 1e-8


class StackedDecomposedRealisticGaussianFilterAdapter(
        RealisticAirVelocityFieldInterface):

    _log: logging.Logger
    _params: StackedDecomposedRealisticGaussianFilterAdapterParameters

    _wrapped_field: RealisticAirVelocityFieldInterface

    def __init__(
            self, wrapped_field: StackedRealisticAirVelocityFieldInterface,
            params: StackedDecomposedRealisticGaussianFilterAdapterParameters):
        self._wrapped_field = wrapped_field
        self._params = params

        self._log = logging.getLogger(__class__.__name__)

    def _query_field(self,
                     X: np.ndarray[Literal["N", 3], float],
                     t: float = 0,
                     include: str | list | np.ndarray = None,
                     exclude: str | list | np.ndarray = None,
                     relative_to_ground: bool = True,
                     return_components: bool = False):

        # query the wrapped field
        wrapped_result = self._wrapped_field.get_velocity(
            X=X,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=return_components)

        velocity = wrapped_result[0] if return_components else wrapped_result
        self._log.debug('velocity shape=%s', velocity.shape)

        return velocity, wrapped_result[1] if return_components else None

    def _find_coordinates_to_filter(self, coords: np.ndarray[Literal["N", 3],
                                                             float],
                                    velocity: np.ndarray[Literal["N", 3],
                                                         float], t_s: float):

        z = coords[:, 2]

        # calculate the distance from the core
        core_position_earth_xy_m = self._wrapped_field.get_thermal_core(z=z,
                                                                        t=t_s)

        distance_from_core_m = np.linalg.norm(coords[:, :2] -
                                              core_position_earth_xy_m,
                                              axis=1)
        distance_from_core_mask = distance_from_core_m <= self._params.maximum_distance_from_core

        # find the segment joints
        segment_relative_altitude = self._wrapped_field.segment_relative_altitude(
            z)
        segment_size = self._wrapped_field.segment_size

        coordinate_mask = (segment_relative_altitude
                           < self._params.segment_bottom_altitude_region_m) | (
                               segment_relative_altitude > segment_size -
                               self._params.segment_top_altitude_region_m)

        # combine the masks
        mask = coordinate_mask & distance_from_core_mask

        # find zero values (if enabled)
        w = velocity[:, 2]
        if self._params.zero_mask_enabled:
            zero_mask = np.abs(w) < self._params.zero_epsilon

            if zero_mask.shape[0] < distance_from_core_m.shape[0]:
                self._log.debug(
                    'zero values\' minimum distance from the core=%s',
                    np.min(distance_from_core_m[zero_mask]))

            mask = mask & zero_mask

        self._log.debug('mask length=%s', mask.sum())

        coords_to_filter = coords[mask]
        self._log.debug('coords_to_filter shape=%s', coords_to_filter.shape)

        return coords_to_filter, mask

    def _query_velocities_for_filter(
        self,
        coords_for_filter_calculation: np.ndarray[Literal["N", 3], float],
        t: float = 0,
        include: str | list | np.ndarray = None,
        exclude: str | list | np.ndarray = None,
        relative_to_ground: bool = True,
    ):

        self._log.debug('calculating unique coordinates for coords.shape=%s',
                        coords_for_filter_calculation.shape)

        # create unique list of coordinates for performance
        unique_coords, inverse_indices = np.unique(
            coords_for_filter_calculation, axis=0, return_inverse=True)
        self._log.debug(
            'querying velocities; unique_coords.shape=%s,inverse_indices.shape=%s',
            unique_coords.shape, inverse_indices.shape)

        # query the values from the original air velocity field
        velocities_for_filter, _ = self._query_field(
            X=unique_coords,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=False)

        all_velocities_for_filter = velocities_for_filter[inverse_indices]
        self._log.debug('velocities queried for filter; shape=%s',
                        all_velocities_for_filter.shape)

        return all_velocities_for_filter

    def _create_coordinate_list_for_filter(
            self, coords_to_filter: np.ndarray[Literal["N", 3], float]):

        t1 = time.perf_counter()
        self._log.debug('creating coordinate list for filter...')
        all_coords = None
        x_size = None
        y_size = None
        z_size = None
        package_size = None
        package_count = coords_to_filter.shape[0]

        size_xyz_m = self._params.filter_size_xyz_m
        spacing_xyz_m = self._params.filter_spacing_xyz_m

        for i in range(package_count):
            coord = coords_to_filter[i]

            # create the space around the coordinate
            x_space = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
                center=coord[0], size=size_xyz_m[0], spacing=spacing_xyz_m[0])
            y_space = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
                center=coord[1], size=size_xyz_m[1], spacing=spacing_xyz_m[1])
            z_space = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
                center=coord[2], size=size_xyz_m[2], spacing=spacing_xyz_m[2])

            if x_size is None:
                # initialize the size
                x_size = len(x_space)
                y_size = len(y_space)
                z_size = len(z_space)
                package_size = x_size * y_size * z_size

            # create coordinate list for this space
            GX, GY, GZ = np.meshgrid(x_space, y_space, z_space, indexing='ij')
            coordinate_list = np.stack([item.ravel() for item in (GX, GY, GZ)],
                                       axis=1)
            if all_coords is None:
                # initialize the list of all coordinates for each filter region
                all_coords = np.empty((package_size * package_count, 3))

            all_coords[i * package_size:i * package_size +
                       package_size, :] = coordinate_list

        t2 = time.perf_counter()
        self._log.debug(
            'coordinate list created for filter; shape=%s,elapsed_time_s=%s',
            all_coords.shape, t2 - t1)

        return all_coords

    def _gaussian_filter(self, coords_to_filter: np.ndarray[Literal["N", 3]],
                         data: np.ndarray[float]):
        # create the kernel
        _, _, _, kernel = self._create_3d_gaussian_kernel(
            size_xyz=self._params.filter_size_xyz_m,
            spacing_xyz=self._params.filter_spacing_xyz_m,
            sigma_xyz=self._params.filter_sigma_xyz_m)

        package_size = np.size(kernel)

        filtered_data = np.zeros((coords_to_filter.shape[0]))
        for i in range(coords_to_filter.shape[0]):

            # reconstruct the grid for the filter
            package_data = data[i * package_size:i * package_size +
                                package_size]
            package_data_for_filter = package_data.reshape(
                (kernel.shape[0], kernel.shape[1], kernel.shape[2]))

            # element-wise multiplication and sum
            value = np.multiply(package_data_for_filter, kernel)
            value = np.sum(value)

            # find the fraction of zeros in the package
            package_data_zero_fraction = np.sum(
                package_data_for_filter < self._params.zero_epsilon) / np.size(
                    package_data_for_filter)

            # set amplitude based on zero fraction in the input data
            value = value * (self._params.amplitude_multiplier +
                             package_data_zero_fraction)

            filtered_data[i] = value

        return filtered_data

    def get_velocity(
        self,
        X: np.ndarray[Literal[3], float] | np.ndarray[Literal["N", 3], float],
        t: float = 0,
        include: str | list | np.ndarray = None,
        exclude: str | list | np.ndarray = None,
        relative_to_ground: bool = True,
        return_components: bool = False,
    ):
        # standarize the input
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape((1, 3))

        # query the field with original coordinates
        velocity, components = self._query_field(
            X=X,
            t=t,
            include=include,
            exclude=exclude,
            relative_to_ground=relative_to_ground,
            return_components=return_components)

        # determine the coordinates which should be filtered
        coords_to_filter, mask = self._find_coordinates_to_filter(
            coords=X, velocity=velocity, t_s=t)

        self._log.debug('coords_to_filter.shape=%s', coords_to_filter.shape)

        if coords_to_filter.shape[0] > 0:

            # create list of cordinates which should be queried for calculating the filter
            coords_for_filter_calculation = self._create_coordinate_list_for_filter(
                coords_to_filter)

            # query the velocities for filter calculation
            all_velocities_for_filter = self._query_velocities_for_filter(
                coords_for_filter_calculation=coords_for_filter_calculation,
                t=t,
                include=include,
                exclude=exclude,
                relative_to_ground=relative_to_ground)

            # execute the gaussian filter for each coordinate
            w_for_filter = all_velocities_for_filter[:, 2]
            filtered_w = self._gaussian_filter(
                coords_to_filter=coords_to_filter, data=w_for_filter)
            self._log.debug('filtered_w.shape=%s', filtered_w.shape)

            # fill the original velocity with the filtered values
            velocity = np.copy(velocity)

            # use the max of the filtered and the original
            max_w = np.max((velocity[mask, 2], filtered_w), axis=0)
            velocity[mask, 2] = max_w

        if return_components:
            return velocity, components
        else:
            return velocity

    @staticmethod
    def _create_3d_gaussian_kernel(size_xyz: tuple[float, float, float],
                                   spacing_xyz: tuple[float, float, float],
                                   sigma_xyz: tuple[int, int, int]):
        # create the grid
        x = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
            center=0., size=size_xyz[0], spacing=spacing_xyz[0])
        y = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
            center=0., size=size_xyz[1], spacing=spacing_xyz[1])
        z = StackedDecomposedRealisticGaussianFilterAdapter.linspace_with_center(
            center=0., size=size_xyz[2], spacing=spacing_xyz[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        mu_xyz = (0., 0., 0.)

        # create the kernel
        a = 1 / (2 * sigma_xyz[0]**2)
        b = 1 / (2 * sigma_xyz[1]**2)
        c = 1 / (2 * sigma_xyz[2]**2)

        kernel = np.exp(-(a * (X - mu_xyz[0])**2 + b * (Y - mu_xyz[1])**2 + c *
                          (Z - mu_xyz[2])**2))

        # normalize
        kernel = kernel / np.sum(kernel)

        return X, Y, Z, kernel

    @staticmethod
    def linspace_with_center(center: float, size: float, spacing: float):
        half_size = size / 2.

        half_space = np.linspace(0, half_size, int(half_size / spacing + 1))
        space = np.hstack(
            (np.flip(center - half_space), (half_space + center)[1:]))
        return space

    def get_thermal_core(self,
                         z: np.ndarray[float] | float,
                         t: float = 0,
                         **kwargs):
        return self._wrapped_field.get_thermal_core(z=z, t=t, **kwargs)

    def info(self):

        wrapped_info = self._wrapped_field.info()

        # remove the filtered region, it is unusable for statistics
        return dict(
            original_thermal_z_min=wrapped_info['original_thermal_z_min'] +
            self._params.segment_bottom_altitude_region_m,
            original_thermal_z_max=wrapped_info['original_thermal_z_max'] -
            self._params.segment_top_altitude_region_m)
