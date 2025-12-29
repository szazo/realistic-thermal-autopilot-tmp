from pathlib import Path
from .bird_aerodynamics_config_generator import BirdAerodynamicsConfigGenerator
from .realistic_single_agent_with_birds_wing_loading_config_generator import (
    RealisticSingleAgentWithBirdsWingLoadingConfigGenerator)
from .realistic_peer_informed_with_birds_wing_loading_config_generator import (
    RealisticPeerInformedWithBirdsWingLoadingConfigGenerator)
from .calculate_initial_and_cutoff_altitude import InitialAndCutoffAltitudeParams
from ruamel.yaml import YAML
from .bird_aerodynamics_loader import BirdAerodynamicsLoader
from .prepare_bird_trajectory_config_generator import PrepareBirdTrajectoryConfigGenerator

from bird_comparison.config_generator.api import (Config, AerodynamicsInfo,
                                                  Thermal)


def create_yaml(config: dict, out_filepath: Path, insert_package_global: bool):

    out_filepath.parent.mkdir(parents=True, exist_ok=True)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096  # prevent line break
    with open(out_filepath, 'w') as f:
        if insert_package_global:
            f.write('# @package _global_\n')
        yaml.dump(config, f)


def save_configs(configs: list[Config], out_dir: Path,
                 insert_package_global: bool):

    for config in configs:

        out_filepath = out_dir / f'{config.name}.yaml'
        create_yaml(config=config.config,
                    out_filepath=out_filepath,
                    insert_package_global=insert_package_global)


def load_bird_wing_loadings(input_filepath: Path, seed: int):
    wing_loading_loader = BirdAerodynamicsLoader()
    bird_aerodynamics = wing_loading_loader.load(input_filepath, seed=seed)

    return bird_aerodynamics


def prepare_bird_trajectories_configs(
        thermal_descriptors: list[Thermal],
        bird_aerodynamics: dict[str, AerodynamicsInfo], out_dir: Path):

    generator = PrepareBirdTrajectoryConfigGenerator(
        bird_aerodynamics=bird_aerodynamics)
    configs = generator.generate(thermal_descriptors=thermal_descriptors)

    save_configs(configs=configs, out_dir=out_dir, insert_package_global=True)


def realistic_single_agent_with_birds_wing_loading_configs(
        thermal_descriptors: list[Thermal],
        params: InitialAndCutoffAltitudeParams, aerodynamics_config_path: Path,
        out_dir: Path):

    generator = RealisticSingleAgentWithBirdsWingLoadingConfigGenerator(
        params=params, aerodynamics_config_path=aerodynamics_config_path)

    configs = generator.generate(thermal_descriptors=thermal_descriptors)
    save_configs(configs=configs, out_dir=out_dir, insert_package_global=True)


def realistic_peer_informed_with_birds_wing_loading_configs(
        thermal_descriptors: list[Thermal],
        params: InitialAndCutoffAltitudeParams, aerodynamics_config_path: Path,
        out_dir: Path):

    generator = RealisticPeerInformedWithBirdsWingLoadingConfigGenerator(
        params=params, aerodynamics_config_path=aerodynamics_config_path)

    configs = generator.generate(thermal_descriptors=thermal_descriptors)
    save_configs(configs=configs, out_dir=out_dir, insert_package_global=True)


def save_bird_aerodynamics_configs(bird_aerodynamics: dict[str,
                                                           AerodynamicsInfo],
                                   out_dir: Path):

    generator = BirdAerodynamicsConfigGenerator()
    configs = generator.generate(bird_aerodynamics=bird_aerodynamics)

    save_configs(configs=configs, out_dir=out_dir, insert_package_global=False)
