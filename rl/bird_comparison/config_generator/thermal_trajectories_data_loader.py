from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .api import (Thermal, ThermalMeta, BirdTrajectoryMeta)


@dataclass
class InputThermalDetails:
    name: str
    stork_trajectory_path: str
    reconstructed_thermal_path: str


class ThermalTrajectoriesDataLoader:

    _root_path: Path
    _trajectory_relative_base_path: Path

    def __init__(self, root_path: Path, trajectory_relative_base_path: Path):

        self._root_path = root_path
        self._trajectory_relative_base_path = trajectory_relative_base_path

    def load(self, input: list[InputThermalDetails]) -> list[Thermal]:

        thermal_descriptors: list[Thermal] = []
        for thermal_input in input:
            thermal = self._process_thermal(thermal_input=thermal_input)
            thermal_descriptors.append(thermal)

        return thermal_descriptors

    def _process_thermal(self, thermal_input: InputThermalDetails) -> Thermal:

        trajectory_relative_path = self._trajectory_relative_base_path / thermal_input.stork_trajectory_path / 'data.csv'
        trajectory_df = pd.read_csv(self._root_path / trajectory_relative_path)

        # group by birds
        bird_groupby = trajectory_df.groupby(by='bird_name')

        bird_descriptors: list[BirdTrajectoryMeta] = []

        for bird_name, bird_df in bird_groupby:

            bird_minimum_altitude_m = float(
                np.array(pd.to_numeric(bird_df['Z'])).min())
            bird_maximum_altitude_m = float(
                np.array(pd.to_numeric(bird_df['Z'])).max())

            assert isinstance(bird_name, str)
            bird_descriptor = BirdTrajectoryMeta(
                name=bird_name,
                minimum_altitude_m=bird_minimum_altitude_m,
                maximum_altitude_m=bird_maximum_altitude_m)
            bird_descriptors.append(bird_descriptor)

        thermal_start_time_s: float = float(
            np.array(pd.to_numeric(trajectory_df['time'])).min())

        thermal_descriptor = Thermal(
            birds=bird_descriptors,
            meta=ThermalMeta(name=thermal_input.name,
                             start_time_s=thermal_start_time_s,
                             trajectory_relative_path=trajectory_relative_path,
                             reconstructed_thermal_path=Path(
                                 thermal_input.reconstructed_thermal_path)))

        return thermal_descriptor
