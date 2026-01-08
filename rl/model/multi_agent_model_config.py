import logging
from functools import partial
from dataclasses import dataclass
from trainer.common.load_policy import PolicyCheckpoint, load_policy
from trainer.multi_agent.parallel_multi_agent_policy_manager import ParallelMultiAgentPolicyManager
from omegaconf import MISSING
import torch
import tianshou
import gymnasium
from hydra.core.config_store import ConfigStore
from trainer.common import (TianshouModelConfigBase)


@dataclass
class NonLearningModelConfig(TianshouModelConfigBase):

    policy_checkpoint: PolicyCheckpoint = MISSING
    device: str = 'cpu'
    _target_: str = 'model.multi_agent_model_config.load_non_learning_model'


def load_non_learning_model(device: str, policy_checkpoint: PolicyCheckpoint,
                            observation_space: gymnasium.Space,
                            action_space: gymnasium.Space,
                            deterministic_eval: bool):

    _log = logging.getLogger('load_non_learning_model')

    _log.debug('loading model...')

    # create the policy
    policy_partial, weights_state_dict = load_policy(source=policy_checkpoint,
                                                     device=device)

    policy = policy_partial(observation_space=observation_space,
                            action_space=action_space,
                            deterministic_eval=deterministic_eval)

    policy.load_state_dict(weights_state_dict)

    _log.debug('model loaded')

    # no learning for this agent
    policy.eval()

    return policy


@dataclass(kw_only=True)
class MultiAgentModelConfig(TianshouModelConfigBase):

    non_learning: NonLearningModelConfig
    learning: TianshouModelConfigBase
    learning_agent_ids: list[str]
    agent_observation_preprocessors: dict[str, str]
    default_action_for_missing_policy: float
    deterministic_eval: bool | None = None
    _target_: str = 'model.multi_agent_model_config.create_multi_agent_model'


def create_multi_agent_model(non_learning: partial[tianshou.policy.BasePolicy],
                             learning: partial[tianshou.policy.BasePolicy],
                             learning_agent_ids: list[str],
                             agent_observation_preprocessors: dict[str, str],
                             device: torch.device,
                             observation_space: gymnasium.Space,
                             action_space: gymnasium.Space,
                             default_action_for_missing_policy: float = 0.,
                             deterministic_eval: bool | None = None):

    single_action_space = next(iter(action_space.values()))

    learning_policy = learning(device=device,
                               observation_space=observation_space,
                               action_space=single_action_space,
                               deterministic_eval=deterministic_eval)

    non_learning_policy = non_learning(observation_space=observation_space,
                                       action_space=single_action_space,
                                       deterministic_eval=deterministic_eval)

    possible_agents = list(observation_space.keys())

    policies = {}
    for agent_id in possible_agents:

        if agent_id in learning_agent_ids:
            policy = learning_policy
        elif agent_id.startswith('teacher'):
            # HACK: we use non learning policy only for teachers
            # TODO: we need to create detailed policy configuration for agents
            policy = non_learning_policy
        else:
            policy = None

        if policy is not None:
            policies[agent_id] = policy

    # create the multi agent policy manager
    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=policies,
        learn_agent_ids=learning_agent_ids,
        agent_observation_preprocessors=agent_observation_preprocessors,
        default_action_for_missing_policy=default_action_for_missing_policy)

    return multi_agent_policy


def register_multi_agent_model_config_groups(base_group: str,
                                             config_store: ConfigStore):
    config_store.store(group=f'{base_group}',
                       name='multi_agent',
                       node=MultiAgentModelConfig)
