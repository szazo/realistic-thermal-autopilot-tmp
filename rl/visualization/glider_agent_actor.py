from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from importlib import resources
import logging
import numpy as np
import quaternion
import vedo

from utils import Vector3, VectorN, VectorNx3
from utils.differential_rotator import DifferentialRotator
from utils.trajectory_state_interpolator import (
    TrajectoryStateInterpolator, TrajectoryEulerInterpolatorInput,
    euler_interpolator_input_to_quaternion)

from .trail_line import TrailLine

ActorState = Literal['not_started'] | Literal['running'] | Literal[
    'cooldown'] | Literal['finished']


@dataclass
class AgentParams:
    agent_type: str
    agent_name: str
    agent_training: str


@dataclass
class Trajectory:
    time_s: VectorN
    position_earth_xyz_m: VectorNx3
    velocity_earth_xyz_m_per_s: VectorNx3
    yaw_pitch_roll_body_to_earth_rad: VectorNx3
    air_velocity_earth_xyz_m_per_s: VectorNx3


@dataclass
class AgentEpisode:
    agent: AgentParams
    trajectory: Trajectory


@dataclass
class MeshParams:
    path: Path
    scale: float


@dataclass
class TrailParams:
    length_s: float
    min_interval_s: float
    linewidth: int
    alpha_min: float
    alpha_max: float


@dataclass
class VelocityArrowParams:
    length_scale: float
    size: float | None


@dataclass
class FlagpostParams:
    show_agent_name: bool


@dataclass
class AgentDisplayParams:
    color: np.ndarray | None
    flagpost: FlagpostParams | None
    velocity_arrow: VelocityArrowParams | None
    trail: TrailParams | None
    mesh: MeshParams


class GliderAgentActor:

    _log: logging.Logger

    _episode: AgentEpisode
    _display_params: AgentDisplayParams

    _current_state: ActorState

    _plane: vedo.Mesh | None
    _flagpost: vedo.Flagpost | None
    _velocity_arrow: vedo.Arrow | None

    _plane_rotator: DifferentialRotator

    _trail_line_min_interval_s: float
    _trail_line: TrailLine | None
    _trail_line_last_frame_index: int

    _interpolator: TrajectoryStateInterpolator

    def __init__(self, episode: AgentEpisode,
                 display_params: AgentDisplayParams):

        self._log = logging.getLogger(__class__.__name__)

        self._episode = episode
        self._display_params = display_params

        self._current_state = 'not_started'

        self._plane_rotator = DifferentialRotator()

        self._plane = None
        self._flagpost = None
        self._velocity_arrow = None

        self._trail_line_last_frame_index = -1
        if display_params.trail is not None:
            trail_params = display_params.trail
            self._trail_line_min_interval_s = trail_params.min_interval_s
            self._trail_line = TrailLine(
                trail_length=trail_params.length_s,
                linewidth=trail_params.linewidth,
                color=display_params.color
                if display_params.color is not None else 'yellow',
                alpha_min=trail_params.alpha_min,
                alpha_max=trail_params.alpha_max)

        self._interpolator = self._create_interpolator(episode.trajectory)

        self._trail_line_last_time_s = None

    def _create_interpolator(self, trajectory: Trajectory):

        euler_input = TrajectoryEulerInterpolatorInput(
            time=trajectory.time_s,
            position_xyz=trajectory.position_earth_xyz_m,
            orientation_yaw_pitch_roll_rad=trajectory.
            yaw_pitch_roll_body_to_earth_rad,
            velocity_xyz=trajectory.velocity_earth_xyz_m_per_s)

        interpolator_input = euler_interpolator_input_to_quaternion(
            euler_input, rotation_type='extrinsic')

        interpolator = TrajectoryStateInterpolator(
            input=interpolator_input,
            position_velocity_interpolation_type='cubic')

        return interpolator

    def step(self, current_time_s: float, plotter: vedo.Plotter) -> ActorState:

        self._log.debug('step; current_time_s=%f', current_time_s)

        # find the smallest frame index for the current time
        frame_times = self._episode.trajectory.time_s

        if current_time_s < frame_times[0]:
            # not yet started
            assert self._current_state == 'not_started'
            return self._current_state
        elif current_time_s > frame_times[-1]:
            # finished, remove from the scene
            if self._current_state == 'running':

                if self._trail_line is not None:
                    self._log.debug(
                        'agent finished, cooling down; current_time_s=%f',
                        current_time_s)
                    self._current_state = 'cooldown'
                    self._destroy(plotter, destroy_trail_line=False)
                else:
                    self._log.debug(
                        'agent finished, removing; current_time_s=%f',
                        current_time_s)
                    self._current_state = 'finished'
                    self._destroy(plotter)
                    return self._current_state
            elif self._current_state == 'cooldown':
                # NOP
                pass
            else:
                self._current_state = 'finished'
                return self._current_state

        if self._current_state == 'not_started':

            self._log.debug('agent started; current_time_s=%f', current_time_s)
            self._create(plotter)
            self._current_state = 'running'

        if self._current_state == 'cooldown':
            assert self._trail_line is not None
            trail_state = self._trail_line.update_trail(index=current_time_s,
                                                        plotter=plotter)
            if trail_state == 'empty':
                self._current_state = 'finished'
                self._trail_line.destroy(plotter)
        else:
            # update
            self._update(current_time_s=current_time_s, plotter=plotter)

        return self._current_state

    def _update(self, current_time_s: float, plotter: vedo.Plotter):

        self._log.debug('_update; current_time_s=%d', current_time_s)

        state = self._interpolator.query(current_time_s)

        if state == None:
            return False

        self._update_from_pose_and_velocity(
            current_time_s=current_time_s,
            current_position=state.position_xyz,
            current_velocity=state.velocity_xyz,
            current_rotation=state.orientation,
            plotter=plotter)

        return True

    def _update_from_pose_and_velocity(self, current_time_s: float,
                                       current_position: Vector3,
                                       current_velocity: Vector3,
                                       current_rotation: quaternion.quaternion,
                                       plotter: vedo.Plotter):

        self._update_plane(position_body_to_enu=current_position,
                           orientation_body_to_enu=current_rotation)

        if self._trail_line is not None:

            if self._trail_line_last_time_s is None or current_time_s - self._trail_line_last_time_s > self._trail_line_min_interval_s:
                self._trail_line.add_point(index=current_time_s,
                                           pt=current_position.tolist(),
                                           plotter=plotter)
                self._trail_line_last_time_s = current_time_s

        if self._flagpost is not None:
            self._update_flagpost(position=current_position,
                                  velocity=current_velocity)

        if self._display_params.velocity_arrow is not None:
            self._update_velocity_arrow(position=current_position,
                                        velocity=current_velocity,
                                        plotter=plotter)

    def _update_plane(self, position_body_to_enu: Vector3,
                      orientation_body_to_enu: quaternion.quaternion):
        assert self._plane is not None

        # calculate the differential rotation based on the target orientation
        axis, angle_rad = self._plane_rotator.rotate_to(
            orientation_body_to_enu)

        around = (position_body_to_enu[0], position_body_to_enu[1],
                  position_body_to_enu[2])

        self._plane.pos(position_body_to_enu[0], position_body_to_enu[1],
                        position_body_to_enu[2])
        self._plane.rotate(angle=angle_rad, axis=axis, point=around, rad=True)

        self._plane.update_shadows()

    def _update_flagpost(self, position: Vector3, velocity: Vector3):

        assert self._flagpost is not None
        assert self._display_params.flagpost is not None
        self._flagpost.pos((position[0], position[1], position[2]))

        agent_name = ''
        if self._display_params.flagpost.show_agent_name:
            agent_name = f'{self._episode.agent.agent_name}\n'

        txt = f'{agent_name}' + \
            f'alt: {position[2]:.1f} m\n' + \
            f'v_z: {velocity[2]:.1f} m/s'

        self._flagpost.text(txt)

    def _update_velocity_arrow(self, position: Vector3, velocity: Vector3,
                               plotter: vedo.Plotter):

        assert self._display_params.velocity_arrow is not None

        velocity_arrow_start = position
        velocity_arrow_end = position + velocity * self._display_params.velocity_arrow.length_scale
        if self._velocity_arrow is not None:
            plotter.remove(self._velocity_arrow)
        color = 'yellow'
        if self._display_params.color is not None:
            color = self._display_params.color

        self._velocity_arrow = vedo.Arrow(
            start_pt=(velocity_arrow_start[0], velocity_arrow_start[1],
                      velocity_arrow_start[2]),
            end_pt=(velocity_arrow_end[0], velocity_arrow_end[1],
                    velocity_arrow_end[2]),
            s=self._display_params.velocity_arrow.size,
            c=color)
        plotter.add(self._velocity_arrow)

    def _create(self, plotter: vedo.Plotter):

        self._plane_rotator.reset()

        self._plane = self._create_plane()
        plotter.add(self._plane)

        if self._display_params.flagpost is not None:
            self._flagpost = self._create_flagpost(self._plane)
            plotter.add(self._flagpost)

    def _create_plane(self):

        object_url = resources.files('assets.player') / str(
            self._display_params.mesh.path)
        plane = vedo.Mesh(str(object_url))

        plane_color = 'yellow'
        if self._display_params.color is not None:
            plane_color = self._display_params.color

        plane.color(plane_color)
        plane.scale(self._display_params.mesh.scale)

        plane.add_shadow('z', 0)

        return plane

    def _create_flagpost(self, plane: vedo.Mesh):

        text_size = 0.6

        flagpost = plane.flagpost(
            f"",
            offset=(0., 0., 30.),
            alpha=0.5,
            c=(0, 0, 255),
            bc=(255, 0, 255),
            lw=1,
            vspacing=1.2,
            s=text_size,
            font='VictorMono',
        )
        assert flagpost is not None

        return flagpost

    def _destroy(self, plotter: vedo.Plotter, destroy_trail_line: bool = True):

        plotter.remove(self._plane)
        self._plane = None

        if self._flagpost is not None:
            plotter.remove(self._flagpost)
            self._flagpost = None

        if self._velocity_arrow is not None:
            plotter.remove(self._velocity_arrow)
            self._velocity_arrow = None

        if destroy_trail_line and self._trail_line is not None:
            self._trail_line.destroy(plotter)

    def episode(self) -> AgentEpisode:
        return self._episode
