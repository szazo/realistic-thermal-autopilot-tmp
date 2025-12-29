from hydra.core.config_store import ConfigStore
from .distribution import NormalDistributionConfig


def register_distribution_config_groups(config_store: ConfigStore):
    config_store.store(group='distribution',
                       name='normal',
                       node=NormalDistributionConfig)
