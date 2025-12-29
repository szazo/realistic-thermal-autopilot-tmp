import numpy as np
import logging
from ..api import AirVelocityFieldInterface
from .realistic_air_velocity_field_interface import RealisticAirVelocityFieldInterface


class RealisticAirVelocityFieldAdapter(AirVelocityFieldInterface):

    _log: logging.Logger

    _wrapped_field: RealisticAirVelocityFieldInterface
    _name: str

    def __init__(self,
                 wrapped_field: RealisticAirVelocityFieldInterface,
                 name: str = 'unknown_realistic'):

        self._wrapped_field = wrapped_field
        self._name = name

        self._log = logging.getLogger(__class__.__name__)

    def reset(self):
        self._log.debug("reset")

        return self._wrapped_field.info()

    def seed(self, seed: int):
        self._log.debug("seed=%s", seed)

    def get_velocity(
        self,
        x_earth_m: float | np.ndarray,
        y_earth_m: float | np.ndarray,
        z_earth_m: float | np.ndarray,
        t_s: float | np.ndarray,
    ):

        self._log.debug(
            "get_velocity; x_earth_m=%s,y_earth_m=%s,z_earth_m=%s,t_s=%s",
            x_earth_m,
            y_earth_m,
            z_earth_m,
            t_s,
        )

        input_dim = np.max([
            np.array(x_earth_m).ndim,
            np.array(y_earth_m).ndim,
            np.array(z_earth_m).ndim
        ])

        if input_dim > 1:
            # vectorize the coordinates then ravel them into a coordinate list
            vc = self._vectorized_coords(x_earth_m=x_earth_m,
                                         y_earth_m=y_earth_m,
                                         z_earth_m=z_earth_m)
            coords = np.stack([dim_grid.ravel() for dim_grid in vc], axis=1)
        else:
            # simply stack the vectors to a coordinate list
            coords = np.column_stack((np.array(x_earth_m), np.array(y_earth_m),
                                      np.array(z_earth_m)))

        self._log.debug("processed coords=%s", coords)

        velocity, components = self._wrapped_field.get_velocity(
            X=coords, t=t_s, return_components=True)

        if input_dim == 0:
            # wrapped always returns 2 dimensional vector,
            # we return one dimensional vector if we have constant input
            velocity = np.squeeze(velocity, axis=0)

        self._log.debug("velocity=%s,shape=%s", velocity, velocity.shape)

        if input_dim > 1:
            # reshape to match input meshgrid shape
            out_shape = vc[0].shape
            rx = np.reshape(velocity[:, 0], newshape=out_shape)
            ry = np.reshape(velocity[:, 1], newshape=out_shape)
            rz = np.reshape(velocity[:, 2], newshape=out_shape)

            # warn: components are not in meshgrid format
            return np.array([rx, ry, rz]), components
        elif input_dim == 1:
            # the u,v,w components should be the first dimension
            velocity = velocity.T
            return velocity, components

        return velocity, components

    @staticmethod
    @np.vectorize
    def _vectorized_coords(x_earth_m: float, y_earth_m: float,
                           z_earth_m: float):

        return x_earth_m, y_earth_m, z_earth_m

    def get_thermal_core(self, z_earth_m: float | np.ndarray,
                         t_s: float | np.ndarray):

        self._log.debug("get_thermal_core; z_earth_m=%s,t_s=%s", z_earth_m,
                        t_s)

        input_dim = np.array(z_earth_m).ndim

        core_xy_m = self._wrapped_field.get_thermal_core(z=z_earth_m, t=t_s)
        self._log.debug("core_xy=%s,shape=%s", core_xy_m, core_xy_m.shape)

        if input_dim == 0:
            core_xy_m = np.squeeze(core_xy_m, axis=0)

        return core_xy_m

    @property
    def name(self):
        return self._name
