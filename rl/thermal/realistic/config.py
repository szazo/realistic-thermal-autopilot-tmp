from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from .stacked_decomposed_realistic_air_velocity_field_params import (
    StackedDecomposedRealisticAirVelocityFieldParameters)
from ..api import AirVelocityFieldConfigBase


@dataclass(kw_only=True)
class StackedDecomposedRealisticAirVelocityFieldConfig(
        AirVelocityFieldConfigBase):
    params: StackedDecomposedRealisticAirVelocityFieldParameters
    _target_: str = 'thermal.realistic.make_stacked_decomposed_realistic_air_velocity_field.make_stacked_decomposed_realistic_air_velocity_field'


def register_realistic_air_velocity_field_config_groups(
        group: str, config_store: ConfigStore):

    config_store.store(group=group,
                       name='base_stacked_decomposed_realistic',
                       node=StackedDecomposedRealisticAirVelocityFieldConfig)
