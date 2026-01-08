from typing import Any, Literal
from pathlib import Path
from functools import partial
from dataclasses import dataclass
import json
from tianshou.policy import BasePolicy
import torch
import tianshou
import hydra

from .experiment_logger_weight_store import load_torch_weights
from .api import TianshouModelConfigBase
from utils import find_suitable_torch_device


@dataclass
class PolicyCheckpoint:
    path: str
    checkpoint: str


def load_policy(source: PolicyCheckpoint | dict,
                device: Literal['max_memory_cuda'] | str):

    if isinstance(source, PolicyCheckpoint):
        path = source.path
        checkpoint = source.checkpoint
    else:
        path = source['path']
        checkpoint = source['checkpoint']

    base_path = Path(path)

    return _load_policy_from_file(base_path / 'model.json',
                                  base_path / f'{checkpoint}.pth',
                                  device=device)


def _load_policy_from_file(
    model_params_path: Path, weights_path: Path,
    device: Literal['max_memory_cuda'] | str
) -> tuple[partial[BasePolicy], dict[str, Any]]:

    # load the parameters
    with open(model_params_path, 'r') as f:
        model_parameters = json.load(f)

    found_device = find_suitable_torch_device(device)
    policy = _create_policy_model(policy_config=model_parameters,
                                  device=found_device)

    # load weights
    weights = load_torch_weights(path=weights_path, map_location=found_device)

    return policy, weights


def _create_policy_model(
    device: torch.device, policy_config: TianshouModelConfigBase
) -> partial[tianshou.policy.BasePolicy]:

    policy = hydra.utils.instantiate(policy_config, _convert_='object')
    policy = partial(policy, device=device)

    return policy
