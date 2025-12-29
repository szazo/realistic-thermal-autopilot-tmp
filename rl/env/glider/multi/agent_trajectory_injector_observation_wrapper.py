from typing import OrderedDict, TypeVar, Generic, cast
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from icecream import ic

import pettingzoo
from pettingzoo.utils import BaseParallelWrapper

from .agent_trajectory_injector import (AgentTimeScheduleParameters,
                                        AgentTrajectoryInjector,
                                        AgentTrajectoryInjectorFieldMapping,
                                        AgentTrajectoryInjectorParameters,
                                        Observation, AgentResult, QueryResult)

AgentIDType = TypeVar('AgentIDType', bound=str)
ObsType = TypeVar('ObsType', bound=OrderedDict)
ActType = TypeVar('ActType')


@dataclass
class AgentTrajectoryInjectorObservationWrapperParameters:
    trajectory_path: str
    filters: dict[str, str | float | int]
    field_mapping: AgentTrajectoryInjectorFieldMapping
    agent_schedule: AgentTimeScheduleParameters | None


class AgentTrajectoryInjectorObservationWrapper(
        BaseParallelWrapper[AgentIDType, ObsType,
                            ActType], Generic[AgentIDType, ObsType, ActType]):

    _injector: AgentTrajectoryInjector[AgentIDType]

    _prev_native_agents: list[AgentIDType]
    _prev_injected_query: QueryResult | None

    def __init__(self, env: pettingzoo.ParallelEnv[AgentIDType, ObsType,
                                                   ActType],
                 params: AgentTrajectoryInjectorObservationWrapperParameters):

        super().__init__(env)

        df = pd.read_csv(params.trajectory_path)

        # filter the df
        for key, value in params.filters.items():
            df = df.loc[df[key] == value]

        injector_params = AgentTrajectoryInjectorParameters(
            field_mapping=params.field_mapping,
            agent_schedule=params.agent_schedule)
        injector = AgentTrajectoryInjector(params=injector_params)
        injector.load(df)
        self._injector = injector

        possible_agents = self.possible_agents + injector.possible_agents
        self.possible_agents = possible_agents
        self.agents = []
        ic(self.possible_agents)

        self._prev_native_agents = []
        self._prev_injected_query = None

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)

        time_s = self._get_time_from_obs(obs)

        # query trajectory
        injected_query_result = self._injector.query(time_s=time_s)

        # create observation and info
        injected_obs, injected_info = self._get_injected_observations_and_info(
            injected_query_result.agent_results, time_s)

        # merge
        result_obs = {**obs, **injected_obs}
        result_info = {**info, **injected_info}

        # save state
        self._save_state(injected_query_result)

        return result_obs, result_info

    def step(
        self, actions: dict[AgentIDType, ActType]
    ) -> tuple[
            dict[AgentIDType, ObsType],
            dict[AgentIDType, float],
            dict[AgentIDType, bool],
            dict[AgentIDType, bool],
            dict[AgentIDType, dict],
    ]:

        # filter keys for the wrapped env
        native_actions = {}
        for agent_id in self._prev_native_agents:
            native_actions[agent_id] = actions[agent_id]

        obs, reward, terminated, truncated, info = self.env.step(
            native_actions)

        # if the wrapped agents done, we also truncate the injected trajectories
        is_native_finished = len(self.env.agents) == 0

        time_s = self._get_time_from_obs(obs)

        # query trajectory
        injected_query_result = self._injector.query(time_s=time_s)

        # if there is no observation for an injected agent, but there was in the previous, we use
        # the previous and truncate the agent
        assert self._prev_injected_query is not None
        new_injected_agents = set(injected_query_result.agents)

        missing_agent_ids: list[AgentIDType] = []
        missing_agent_results: dict[AgentIDType, AgentResult] = {}
        for agent_id in self._prev_injected_query.agents:
            if agent_id not in new_injected_agents:
                missing_agent_ids.append(agent_id)
                missing_agent_results[
                    agent_id] = self._prev_injected_query.agent_results[
                        agent_id]

        # create observation and info
        injected_obs, injected_info = self._get_injected_observations_and_info(
            injected_query_result.agent_results, time_s)

        missing_obs, missing_info = self._get_injected_observations_and_info(
            missing_agent_results, time_s)

        injected_reward, injected_terminated, injected_truncated = self._create_reward_terminated_truncated(
            injected_query_result.agents, target_truncated=is_native_finished)
        missing_reward, missing_terminated, missing_truncated = self._create_reward_terminated_truncated(
            missing_agent_ids, target_truncated=True)

        # merge
        result_obs = {**obs, **injected_obs, **missing_obs}
        result_info = {**info, **injected_info, **missing_info}

        # create reward
        result_reward = {**reward, **injected_reward, **missing_reward}
        result_terminated = {
            **terminated,
            **injected_terminated,
            **missing_terminated
        }
        result_truncated = {
            **truncated,
            **injected_truncated,
            **missing_truncated
        }

        # save state
        if is_native_finished:
            self.agents = []
            self._prev_native_agents = []
            self._prev_injected_query = None
        else:
            self._save_state(injected_query_result)

        return result_obs, result_reward, result_terminated, result_truncated, result_info

    def _save_state(self, injected_query_result: QueryResult):
        # create agent list
        native_agents = self.env.agents
        injected_agents = injected_query_result.agents
        self.agents = native_agents + injected_agents

        self._prev_native_agents = list(native_agents)
        self._prev_injected_query = injected_query_result

    def _create_reward_terminated_truncated(self, agent_ids: list[AgentIDType],
                                            target_truncated: bool):

        reward: dict[AgentIDType, float] = {}
        terminated: dict[AgentIDType, bool] = {}
        truncated: dict[AgentIDType, bool] = {}

        for agent_id in agent_ids:
            reward[agent_id] = 0.
            terminated[agent_id] = False
            truncated[agent_id] = target_truncated

        return reward, terminated, truncated

    def _get_time_from_obs(self, obs: dict[AgentIDType, ObsType]) -> float:
        first_obs = obs[next(iter(obs))]
        time_s = float(first_obs['t_s'])

        return time_s

    def _get_injected_observations_and_info(
        self, trajectories: dict[AgentIDType, AgentResult], time_s: float
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs: dict[AgentIDType, ObsType] = {}
        info: dict[AgentIDType, dict] = {}

        for agent_id, agent_result in trajectories.items():

            # create the obs
            agent_obs = agent_result.observation
            agent_obs_dict = OrderedDict(
                position_earth_xyz_m=agent_obs.position_earth_m_xyz,
                velocity_earth_xyz_m_per_s=agent_obs.
                velocity_earth_m_per_s_xyz,
                yaw_pitch_roll_earth_to_body_rad=agent_obs.
                yaw_pitch_roll_earth_to_body_rad,
                t_s=time_s)

            obs[agent_id] = cast(ObsType, agent_obs_dict)

            # create the info
            agent_info_dict = {**cast(dict, agent_obs_dict)}
            info[agent_id] = agent_info_dict

        return obs, info
