from abc import ABC, abstractmethod
from typing import Literal, TypedDict
import numpy as np

from utils import Vector3, VectorNx3, VectorN


# API for the realistic thermal interface
class RealisticAirVelocityFieldInterface(ABC):

    @abstractmethod
    def get_velocity(
        self,
        X: Vector3 | VectorNx3,
        t: float = 0,
        include: str | list | np.ndarray | None = None,
        exclude: str | list | np.ndarray | None = None,
        relative_to_ground: bool = True,
        return_components: bool = False
    ) -> (Vector3 | VectorNx3) | tuple[(Vector3 | VectorNx3), dict]:
        pass

    @abstractmethod
    def get_thermal_core(self, z: VectorN | float, t: float = 0, **kwargs):
        pass

    def info(self) -> dict:
        return dict()


class UnivariateSplineWrapper(ABC):

    @property
    @abstractmethod
    def x_min(self) -> float:
        pass

    @property
    @abstractmethod
    def x_max(self) -> float:
        pass


class DecomposedThermalCoreSpline(TypedDict):
    X: UnivariateSplineWrapper
    Y: UnivariateSplineWrapper


class DecomposedRealisticAirVelocityFieldInterface(
        RealisticAirVelocityFieldInterface):

    @property
    @abstractmethod
    def current_thermal_core_spline(self) -> DecomposedThermalCoreSpline:
        pass


class StackedRealisticAirVelocityFieldInterface(
        RealisticAirVelocityFieldInterface):

    @property
    @abstractmethod
    def segment_size(self) -> float:
        pass

    @abstractmethod
    def segment_relative_altitude(
            self, z: float | np.ndarray[float]) -> float | np.ndarray[float]:
        pass
