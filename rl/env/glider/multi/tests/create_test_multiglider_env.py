from dataclasses import dataclass, asdict
from env.glider.multi.make_multiglider_env import MultiGliderEnvParameters
from env.glider.base import AgentID, DiscreteToContinuousDegreesParams, SimulationBoxParameters, TimeParameters
from env.glider.base.agent import (GliderCutoffParameters,
                                   GliderInitialConditionsParameters,
                                   GliderRewardParameters,
                                   GliderAgentParameters)
from env.glider.base.visualization import RenderParameters
from env.glider.multi import (AgentGroupParams, AgentSpawnParameters2,
                              MultiGliderEnvParameters, make_multiglider_env)
from thermal.zero import ZeroAirVelocityField
from env.glider.aerodynamics import SimpleAerodynamicsParameters, SimpleAerodynamics
from trainer.multi_agent.parallel_pettingzoo_env import ParallelPettingZooEnv


@dataclass
class Instances:
    env: ParallelPettingZooEnv
    aerodynamics: SimpleAerodynamics
    air_velocity_field: ZeroAirVelocityField


def create_test_multiglider_env_params(
        sequence_length: int = 2,
        max_closest_agent_count: int = 3) -> MultiGliderEnvParameters:

    simulation_box_params = SimulationBoxParameters(box_size=(5000, 5000,
                                                              2000))
    time_params = TimeParameters(dt_s=0.5, decision_dt_s=1.0)
    render_params = RenderParameters(mode='rgb_array')

    agent_params = GliderAgentParameters()

    cutoff_params = GliderCutoffParameters()
    reward_params = GliderRewardParameters()

    initial_conditions_params = GliderInitialConditionsParameters()

    agent_group_params = {
        'teacher':
        AgentGroupParams(
            order=0,
            terminate_if_finished=False,
            initial_conditions_params=initial_conditions_params,
            spawner=AgentSpawnParameters2(pool_size=10,
                                          initial_time_offset_s_min_max=(0, 0),
                                          parallel_num_min_max=(3, 8),
                                          time_between_spawns_min_max_s=(1, 5),
                                          must_spawn_if_no_global_agent=True),
        ),
        'student':
        AgentGroupParams(order=1,
                         terminate_if_finished=True,
                         initial_conditions_params=initial_conditions_params,
                         spawner=AgentSpawnParameters2(
                             pool_size=1,
                             initial_time_offset_s_min_max=(0, 0),
                             parallel_num_min_max=(1, 1),
                             time_between_spawns_min_max_s=(1, 1),
                             must_spawn_if_no_global_agent=False)),
    }

    params = MultiGliderEnvParameters(
        simulation_box_params=simulation_box_params,
        time_params=time_params,
        cutoff_params=cutoff_params,
        reward_params=reward_params,
        glider_agent_params=agent_params,
        max_closest_agent_count=max_closest_agent_count,
        agent_groups=agent_group_params,
        render_params=render_params,
        max_sequence_length=sequence_length,
        discrete_continuous_mapping=DiscreteToContinuousDegreesParams())

    return params


def create_test_multiglider_env(params: MultiGliderEnvParameters):

    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    env = make_multiglider_env(env_name='multiglider',
                               params=params,
                               air_velocity_field=air_velocity_field,
                               aerodynamics=aerodynamics)
    return Instances(env, aerodynamics, air_velocity_field)
