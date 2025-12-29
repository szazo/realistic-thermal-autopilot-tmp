from hydra.core.config_store import ConfigStore
from .air_velocity_coordinate_sampler import GridAirVelocitySamplerConfig, PolarAirVelocitySamplerConfig


def register_air_velocity_sampler_config_groups(config_store: ConfigStore):
    config_store.store(group='air_velocity_sampler',
                       name='grid',
                       node=GridAirVelocitySamplerConfig)

    config_store.store(group='air_velocity_sampler',
                       name='polar',
                       node=PolarAirVelocitySamplerConfig)
