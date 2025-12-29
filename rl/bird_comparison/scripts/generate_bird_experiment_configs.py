import os
from dataclasses import dataclass
from pathlib import Path
from bird_comparison.config_generator.generator import (
    load_bird_wing_loadings, prepare_bird_trajectories_configs,
    InitialAndCutoffAltitudeParams, save_bird_aerodynamics_configs,
    realistic_single_agent_with_birds_wing_loading_configs,
    realistic_peer_informed_with_birds_wing_loading_configs)
from bird_comparison.config_generator.thermal_trajectories_data_loader import (
    ThermalTrajectoriesDataLoader, InputThermalDetails)


@dataclass
class PeerInformedConfig:
    using_bird_wing_loadings: InitialAndCutoffAltitudeParams


@dataclass
class GeneratorConfig:
    wing_loading_seed: int
    single_agent: InitialAndCutoffAltitudeParams

    peer_informed_with_birds: PeerInformedConfig


def main():

    generator_config = GeneratorConfig(
        wing_loading_seed=42,
        peer_informed_with_birds=PeerInformedConfig(
            using_bird_wing_loadings=InitialAndCutoffAltitudeParams(
                start_altitude_offset_relative_to_bird_min_m=0.,
                success_altitude_offset_relative_to_bird_max_m=0.,
                fail_altitude_offset_relative_to_start_m=-300)),
        single_agent=InitialAndCutoffAltitudeParams(
            start_altitude_offset_relative_to_bird_min_m=-200,
            success_altitude_offset_relative_to_bird_max_m=200,
            fail_altitude_offset_relative_to_start_m=-300))

    input = [
        InputThermalDetails(name='b010',
                            stork_trajectory_path='b010_0.1',
                            reconstructed_thermal_path='b010_extrapolated'),
        InputThermalDetails(name='b0230',
                            stork_trajectory_path='b023_0.1',
                            reconstructed_thermal_path='b0230_extrapolated'),
        InputThermalDetails(name='b072',
                            stork_trajectory_path='b072_0.1',
                            reconstructed_thermal_path='b072_extrapolated'),
        InputThermalDetails(name='b077',
                            stork_trajectory_path='b077_0.1',
                            reconstructed_thermal_path='b077_extrapolated'),
        InputThermalDetails(name='b112',
                            stork_trajectory_path='b112_0.2',
                            reconstructed_thermal_path='b112_extrapolated'),
        InputThermalDetails(name='b121',
                            stork_trajectory_path='b121_0.1',
                            reconstructed_thermal_path='b121_extrapolated'),
    ]

    root_path = (Path(os.path.dirname(__file__)) / '../../../').resolve()
    output_config_base_path = root_path / 'config/birds'

    # load bird aerodynamics info
    bird_aerodynamics = load_bird_wing_loadings(
        root_path /
        'data/bird_comparison/wing_loading_stork/wing_loadings.csv',
        seed=generator_config.wing_loading_seed)

    # load thermal trajectories
    processed_trajectory_base_path = Path(
        'data/bird_comparison/processed/stork_trajectories')
    loader = ThermalTrajectoriesDataLoader(
        root_path=root_path,
        trajectory_relative_base_path=processed_trajectory_base_path)
    thermal_descriptors = loader.load(input=input)

    # generate prepare bird trajectories configs
    prepare_bird_trajectories_configs(
        thermal_descriptors=thermal_descriptors,
        bird_aerodynamics=bird_aerodynamics,
        out_dir=output_config_base_path /
        'realistic/prepare_bird_trajectories/birds_in_thermals')

    # save aerodynamics configs
    aerodynamics_config_path = Path('realistic/eval/env/aerodynamics')
    save_bird_aerodynamics_configs(bird_aerodynamics=bird_aerodynamics,
                                   out_dir=output_config_base_path /
                                   aerodynamics_config_path)

    # single agent with bird trajectories and wing loadings
    realistic_single_agent_with_birds_wing_loading_configs(
        thermal_descriptors=thermal_descriptors,
        params=generator_config.single_agent,
        aerodynamics_config_path=aerodynamics_config_path,
        out_dir=output_config_base_path /
        'realistic/eval/single_agent/using_bird_wing_loadings')

    # peer informed with bird trajectories and wing loadings
    realistic_peer_informed_with_birds_wing_loading_configs(
        thermal_descriptors=thermal_descriptors,
        params=generator_config.peer_informed_with_birds.
        using_bird_wing_loadings,
        aerodynamics_config_path=aerodynamics_config_path,
        out_dir=output_config_base_path /
        'realistic/eval/peer_informed_with_birds/using_bird_wing_loadings')


if __name__ == '__main__':
    main()
