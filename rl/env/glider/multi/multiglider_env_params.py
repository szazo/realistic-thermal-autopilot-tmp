from dataclasses import dataclass

from omegaconf import MISSING

from ..base import GliderEnvParameters
from ..base.agent import GliderInitialConditionsParameters
from .agent_spawner import AgentSpawnParameters2
from .agent_trajectory_injector_observation_wrapper import AgentTrajectoryInjectorObservationWrapperParameters


@dataclass
class AgentGroupParams:
    order: int
    terminate_if_finished: bool
    spawner: AgentSpawnParameters2
    initial_conditions_params: GliderInitialConditionsParameters


@dataclass
class MultiGliderEnvParameters(GliderEnvParameters):

    agent_groups: dict[str, AgentGroupParams] = MISSING
    max_closest_agent_count: int = 5
    inject_trajectories: AgentTrajectoryInjectorObservationWrapperParameters | None = None
