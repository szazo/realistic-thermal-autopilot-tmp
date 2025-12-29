from .api import (Thermal, ThermalMeta, BirdTrajectoryMeta, Config,
                  AerodynamicsInfo)


class PrepareBirdTrajectoryConfigGenerator:

    _bird_aerodynamics: dict[str, AerodynamicsInfo]

    def __init__(self, bird_aerodynamics: dict[str, AerodynamicsInfo]):

        self._bird_aerodynamics = bird_aerodynamics

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
            aerodynamics = self._bird_aerodynamics[bird.name]

            thermal_bird_config = self._create_thermal_bird_config(
                thermal_meta=thermal.meta,
                bird_meta=bird,
                aerodynamics=aerodynamics)
            configs.append(thermal_bird_config)

        return configs

    def _create_thermal_bird_config(self, thermal_meta: ThermalMeta,
                                    bird_meta: BirdTrajectoryMeta,
                                    aerodynamics: AerodynamicsInfo) -> Config:

        defaults = [{
            'override /env/glider/air_velocity_field@job.air_velocity_field:':
            f'stacked_decomposed_realistic/{thermal_meta.reconstructed_thermal_path}'
        }, '_self_']

        params = {
            'trajectory_path': str(thermal_meta.trajectory_relative_path),
            'additional_columns': {
                'thermal': thermal_meta.name,
                'bird_name': bird_meta.name,
                'mass_kg': aerodynamics.mass_kg,
                'wing_area_m2': aerodynamics.wing_area_m2,
                'CL': aerodynamics.CL,
                'CD': aerodynamics.CD,
                'original_thermal_z_min': bird_meta.minimum_altitude_m,
                'original_thermal_z_max': bird_meta.maximum_altitude_m
            },
            'trajectory_converter': {
                'shift_time_s': thermal_meta.start_time_s,
                'filters': {
                    'bird_name': bird_meta.name
                }
            }
        }

        self_config = {'job': {'params': params}}

        config = {'defaults': defaults, **self_config}

        experiment_id = f'{thermal_meta.name}_{bird_meta.name.lower()}'

        return Config(name=experiment_id, config=config)
