from pathlib import Path
import numpy as np
from .api import (Thermal, ThermalMeta, BirdTrajectoryMeta, Config)
from .calculate_initial_and_cutoff_altitude import (
    calculate_ai_initial_and_cutoff_altitude, InitialAndCutoffAltitudeParams)


class RealisticSingleAgentWithBirdsWingLoadingConfigGenerator:

    _params: InitialAndCutoffAltitudeParams
    _aerodynamics_config_path: Path

    def __init__(self, params: InitialAndCutoffAltitudeParams,
                 aerodynamics_config_path: Path):
        self._params = params
        self._aerodynamics_config_path = aerodynamics_config_path

    def generate(self, thermal_descriptors: list[Thermal]) -> list[Config]:

        configs: list[Config] = []

        for thermal in thermal_descriptors:

            thermal_configs = self._create_thermal_birds_config(
                thermal=thermal)
            configs.extend(thermal_configs)

        return configs

    def _create_thermal_birds_config(self, thermal: Thermal) -> list[Config]:

        configs: list[Config] = []

        for bird in thermal.birds:
            thermal_bird_config = self._create_thermal_bird_config(
                thermal_meta=thermal.meta, bird_meta=bird)
            configs.append(thermal_bird_config)

        return configs

    def _create_thermal_bird_config(self, thermal_meta: ThermalMeta,
                                    bird_meta: BirdTrajectoryMeta) -> Config:

        altitudes = calculate_ai_initial_and_cutoff_altitude(
            params=self._params, bird_meta=bird_meta)

        bird_name_lower = bird_meta.name.lower()

        defaults = [{
            '../initial_conditions@evaluators.main.env.env.params':
            'realistic_single_agent_initial_conditions'
        }, {
            f'/{self._aerodynamics_config_path}@evaluators.main.env.env.aerodynamics':
            bird_name_lower
        }, {
            'override /env/glider/air_velocity_field@evaluators.main.env.env.air_velocity_field':
            f'stacked_decomposed_realistic/{thermal_meta.reconstructed_thermal_path}'
        }]

        experiment_id = f'{thermal_meta.name}_{bird_name_lower}'

        config = {
            'defaults': defaults,
            'experiment_name': experiment_id,
            'evaluators': {
                'main': {
                    'observation_logger': {
                        'additional_columns': {
                            'thermal': thermal_meta.name,
                            'bird_name': bird_meta.name
                        }
                    },
                    'env': {
                        'env': {
                            'params': {
                                'initial_conditions_params': {
                                    'altitude_earth_m_mean':
                                    altitudes.start_altitude_m
                                },
                                'cutoff_params': {
                                    'success_altitude_m':
                                    altitudes.success_altitude_m,
                                    'fail_altitude_m':
                                    altitudes.fail_altitude_m
                                }
                            }
                        }
                    }
                },
            }
        }

        return Config(name=experiment_id, config=config)
