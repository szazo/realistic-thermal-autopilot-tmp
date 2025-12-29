import os
import unittest
from pathlib import Path
from dataclasses import asdict
from typing import OrderedDict, cast
from ..agent_trajectory_injector import AgentTimeScheduleParameters
from env.glider.multi.make_multiglider_env import make_multiglider_env_base
from env.glider.multi.tests.create_test_multiglider_env import create_test_multiglider_env_params
from env.glider.base.agent import (GliderInitialConditionsParameters)
from env.glider.multi import (AgentGroupParams, AgentSpawnParameters2)
from ..agent_trajectory_injector_observation_wrapper import AgentTrajectoryInjectorObservationWrapper, AgentTrajectoryInjectorObservationWrapperParameters
from ..agent_trajectory_injector import (AgentTrajectoryInjectorFieldMapping)
from thermal.zero import ZeroAirVelocityField
from env.glider.aerodynamics import SimpleAerodynamicsParameters, SimpleAerodynamics

from icecream import ic


#@unittest.skip("development sandbox")
def test_init():

    params = create_test_multiglider_env_params()

    # insert only the student
    params.agent_groups = {
        'student':
        AgentGroupParams(
            order=1,
            terminate_if_finished=True,
            initial_conditions_params=GliderInitialConditionsParameters(),
            spawner=AgentSpawnParameters2(pool_size=1,
                                          initial_time_offset_s_min_max=(0, 0),
                                          parallel_num_min_max=(1, 1),
                                          time_between_spawns_min_max_s=(1, 1),
                                          must_spawn_if_no_global_agent=True))
    }

    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    base_env = make_multiglider_env_base(params=params,
                                         air_velocity_field=air_velocity_field,
                                         aerodynamics=aerodynamics)

    base_path = Path(os.path.dirname(__file__)) / '../../../../../'
    input_path = base_path / 'data/bird_comparison/processed/stork_trajectories_as_observation_log/merged_observation_log.csv'

    agent_schedule = AgentTimeScheduleParameters(start_time_s=3., spacing_s=2.)

    field_mapping = AgentTrajectoryInjectorFieldMapping(
        scene_field='thermal',
        agent_name_field='bird_name',
        time_s_field='time_s')

    params = AgentTrajectoryInjectorObservationWrapperParameters(
        trajectory_path=str(input_path),
        field_mapping=field_mapping,
        filters={'thermal': 'b010'},
        agent_schedule=agent_schedule)

    wrapper = AgentTrajectoryInjectorObservationWrapper(base_env, params)
    env = cast(
        AgentTrajectoryInjectorObservationWrapper[str, OrderedDict, float],
        wrapper)

    ic(env.possible_agents)
    ic(env.agents)

    obs, info = env.reset()
    ic(obs, info)
    ic(env.agents)

    actions: dict = {'student0': 0.0}

    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(actions)
        ic(obs, reward, terminated, truncated, env.agents)
