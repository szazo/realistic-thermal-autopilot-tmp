from typing import TypeVar, Generic, cast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils.vector import Vector3
from utils.modular_shortest_path_interpolator import ModularShortestPathInterpolator


@dataclass
class AgentMeta:
    mass_kg: float
    wing_area_m2: float
    CL: float
    CD: float


@dataclass
class Observation:
    position_earth_m_xyz: Vector3
    velocity_earth_m_per_s_xyz: Vector3
    # REVIEW: body_to_earth
    yaw_pitch_roll_earth_to_body_rad: Vector3


class TrajectoryInterpolator:

    _meta: AgentMeta
    _position_velocity_interpolator: interp1d
    _orientation_interpolator: ModularShortestPathInterpolator

    def __init__(self, df: pd.DataFrame):

        # get meta
        first_row = df.iloc[0]
        self._meta = AgentMeta(mass_kg=first_row['mass_kg'],
                               wing_area_m2=first_row['wing_area_m2'],
                               CL=first_row['CL'],
                               CD=first_row['CD'])

        # create the interpolator
        time_s = df['time_s'].to_numpy()
        data_df = df[[
            'position_earth_m_x', 'position_earth_m_y', 'position_earth_m_z',
            'velocity_earth_m_per_s_x', 'velocity_earth_m_per_s_y',
            'velocity_earth_m_per_s_z'
        ]]
        data = data_df.to_numpy()
        self._position_velocity_interpolator = interp1d(time_s,
                                                        data,
                                                        axis=0,
                                                        bounds_error=False,
                                                        fill_value=np.nan)

        orientation_df = df[['yaw_deg', 'pitch_deg', 'roll_deg']]
        orientation_data = orientation_df.to_numpy()
        self._orientation_interpolator = ModularShortestPathInterpolator(
            time_s, orientation_data, low=-180, high=180)

    def query(self, time_s: float) -> Observation | None:

        position_velocity = self._position_velocity_interpolator(time_s)

        if np.all(np.isnan(position_velocity)):
            return None

        orientation = self._orientation_interpolator(time_s)
        assert orientation is not None

        obs = Observation(
            position_earth_m_xyz=np.array([
                position_velocity[0], position_velocity[1],
                position_velocity[2]
            ]),
            velocity_earth_m_per_s_xyz=np.array([
                position_velocity[3], position_velocity[4],
                position_velocity[5]
            ]),
            yaw_pitch_roll_earth_to_body_rad=np.deg2rad(
                np.array([orientation[0], orientation[1], orientation[2]])))

        return obs

    @property
    def meta(self):
        return self._meta


@dataclass
class AgentTimeScheduleParameters:
    start_time_s: float
    spacing_s: float


@dataclass
class AgentTrajectoryInjectorFieldMapping:
    scene_field: str
    agent_name_field: str
    time_s_field: str


@dataclass
class AgentTrajectoryInjectorParameters:
    field_mapping: AgentTrajectoryInjectorFieldMapping
    agent_schedule: AgentTimeScheduleParameters | None


AgentIDType = TypeVar('AgentIDType', bound=str)


@dataclass
class AgentResult:
    observation: Observation
    meta: AgentMeta


@dataclass
class QueryResult(Generic[AgentIDType]):
    agents: list[AgentIDType]
    agent_results: dict[AgentIDType, AgentResult]


class AgentTrajectoryInjector(Generic[AgentIDType]):

    _params: AgentTrajectoryInjectorParameters

    _possible_agents: list[AgentIDType] = []
    _interpolators: dict[str, TrajectoryInterpolator] = {}

    def __init__(self, params: AgentTrajectoryInjectorParameters):
        self._params = params

    def load(self, df: pd.DataFrame):

        if self._params.agent_schedule is not None:
            df = self._shift_agent_times(
                df,
                field_mapping=self._params.field_mapping,
                schedule=self._params.agent_schedule)

        field_mapping = self._params.field_mapping
        agent_groupby = df.groupby(field_mapping.agent_name_field)
        interpolators: dict[str, TrajectoryInterpolator] = {}

        for agent_name, bird_df in agent_groupby:
            agent_name = str(agent_name)

            interpolator = TrajectoryInterpolator(df=bird_df)
            interpolators[agent_name] = interpolator

        self._possible_agents = [
            cast(AgentIDType, key) for key in interpolators.keys()
        ]  #  list(interpolators.keys())
        self._interpolators = interpolators

    def _shift_agent_times(
            self, df: pd.DataFrame,
            field_mapping: AgentTrajectoryInjectorFieldMapping,
            schedule: AgentTimeScheduleParameters) -> pd.DataFrame:

        agent_groupby = df.groupby(
            [field_mapping.scene_field, field_mapping.agent_name_field],
            group_keys=False)

        sorted_agent_names = sorted(list(
            df[field_mapping.agent_name_field].unique()),
                                    key=lambda x: str(x))

        def shift_agent_time(agent_df: pd.DataFrame):

            agent_name = agent_df.name[1]
            agent_index = sorted_agent_names.index(agent_name)

            time_s = np.array(
                agent_df[field_mapping.time_s_field].astype('float32'))

            # shift to zero
            time_s -= time_s.min()

            # shift to specified start
            time_s += schedule.start_time_s + (schedule.spacing_s *
                                               agent_index)

            agent_df[field_mapping.time_s_field] = time_s

            return agent_df

        df = agent_groupby.apply(shift_agent_time)

        return df

    def query(self, time_s: float) -> QueryResult:

        agent_results: dict[AgentIDType, AgentResult] = {}
        agents: list[AgentIDType] = []

        for name, interpolator in self._interpolators.items():

            agent_id = cast(AgentIDType, name)
            obs = interpolator.query(time_s)

            if obs is not None:
                agent_results[agent_id] = AgentResult(observation=obs,
                                                      meta=interpolator.meta)
                agents.append(agent_id)

        return QueryResult(agents=agents, agent_results=agent_results)

    @property
    def possible_agents(self) -> list[AgentIDType]:
        return self._possible_agents
