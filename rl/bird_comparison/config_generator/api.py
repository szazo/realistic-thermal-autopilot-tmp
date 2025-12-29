from typing import Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BirdTrajectoryMeta:
    name: str
    minimum_altitude_m: float
    maximum_altitude_m: float


@dataclass
class ThermalMeta:
    name: str
    start_time_s: float
    reconstructed_thermal_path: Path
    trajectory_relative_path: Path


@dataclass
class Thermal:
    meta: ThermalMeta
    birds: list[BirdTrajectoryMeta]


@dataclass
class Config:
    name: str
    config: dict[str, Any]


@dataclass
class AerodynamicsInfo:
    CL: float
    CD: float
    mass_kg: float
    wing_area_m2: float
