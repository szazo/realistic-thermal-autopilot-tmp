from pathlib import Path
from enum import Enum
from typing import Literal, Any
import time
import logging
from dataclasses import asdict, dataclass
import numpy as np
import vedo

from .air_velocity_scene_actor import SceneParams, Scene, AirVelocitySceneActor

PlaybackFrameSelection = Enum('PlaybackFrameSelection', ['wall_time', 'fixed'])


@dataclass
class CameraParams:
    pos: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    clipping_range: tuple[float, float] | None = None
    roll: float | None = None
    distance: float | None = None
    viewup: tuple[float, float, float] | None = None


@dataclass
class CameraAnimationKeyFrame:
    time_s: float
    camera: CameraParams


@dataclass
class CameraAnimationParams:
    smooth: bool
    keyframes: list[CameraAnimationKeyFrame]


@dataclass
class VideoParams:
    fps: int
    output_path: Path
    backend: str


@dataclass
class CustomAxesParams:
    text_scale: float
    xtick_length: float
    xtick_thickness: float
    ytick_thickness: float
    xtitle: str = 'x'
    ytitle: str = 'y'
    ztitle: str = 'z'


@dataclass
class AxesParams:
    type: int = 1
    custom: CustomAxesParams | None = None


@dataclass(kw_only=True)
class ScenePlayerParams(SceneParams):

    timer_dt_ms: int  # used for scheduling the update

    frame_selection: PlaybackFrameSelection
    wall_time_playback_speed: float | None = None  # used when wall_time is used, time_step_s = deltaT * playback_speed
    fixed_frame_step_s: float | None = None  # used when fixed frame selection is used

    start_scene: int
    start_episode: int
    start_time_s: float

    axes: AxesParams
    video: VideoParams | None = None
    camera: CameraParams | None = None
    camera_animation: CameraAnimationParams | None = None
    hide_button_on_start: bool = False


@dataclass
class CameraAnimation:
    smooth: bool
    cameras: list[dict[str, Any]]
    times: list[float]
    max_time_s: float


PlayerState = Literal['not_started'] | Literal['running'] | Literal['finished']


class ScenePlayer:

    _log: logging.Logger

    _params: ScenePlayerParams
    _scenes: list[Scene]

    _plotter: vedo.Plotter
    _start_stop_button: vedo.Button
    _corner_text: vedo.CornerAnnotation | None

    _timer_callback_id: int | None
    _button_callback_id: int | None
    _timer_id: int | None
    _last_time_s: float | None

    _current_state: PlayerState

    _current_scene_actor: AirVelocitySceneActor | None
    _current_scene_index: int
    _is_show_called: bool

    _video: vedo.Video | None

    _fixed_camera: dict | None
    _camera_animation: CameraAnimation | None

    def __init__(self, params: ScenePlayerParams, scenes: list[Scene]):

        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._scenes = scenes

        self._plotter = self._create_plotter()
        self._initialize_callbacks(self._plotter)

        assert len(scenes) > 0, "minimum one scene is required"

        self._is_show_called = False
        self._current_state = 'not_started'
        self._current_scene_actor = None
        self._video = None

        self._initialize_camera_config()

        if self._params.start_scene >= len(scenes):
            raise ValueError(
                f'invalid start_scene index {self._params.start_scene}; scene count: {len(scenes)}'
            )

        if self._params.frame_selection == PlaybackFrameSelection.wall_time:
            assert self._params.wall_time_playback_speed is not None, 'playback_speed should be defined'
        elif self._params.frame_selection == PlaybackFrameSelection.fixed:
            assert self._params.wall_time_playback_speed is None, 'playback_speed should not be defined if fixed fame selection is used'
            assert self._params.fixed_frame_step_s is not None, 'frame_step_s should be defined'

        self._current_scene_index = self._params.start_scene - 1
        self._create_next_scene(start_episode=params.start_episode,
                                start_time_s=params.start_time_s)

    def _initialize_callbacks(self, plotter: vedo.Plotter):
        self._timer_callback_id = plotter.add_callback('timer',
                                                       self._timer_handler,
                                                       enable_picking=False)
        self._button_callback_id = plotter.add_callback(
            'end interaction',
            self._end_interaction_handler,
            enable_picking=False)
        self._timer_id = None
        self._last_time_s = None

    def _end_interaction_handler(self, event):
        # reset the time after an interaction to prevent time jumping
        self._last_time_s = time.time()

    def _destroy_callbacks(self):
        if self._timer_callback_id is not None:
            self._plotter.remove_callback(self._timer_callback_id)
            self._timer_callback_id = None

        if self._button_callback_id is not None:
            self._plotter.remove_callback(self._button_callback_id)
            self._button_callback_id = None

    def _create_plotter(self):

        plotter = vedo.Plotter()

        plotter.use_depth_peeling(value=True)

        # create start/stop button
        self._start_stop_button = self._create_start_stop_button(plotter)

        self._corner_text = vedo.CornerAnnotation()

        plotter.add(self._start_stop_button, self._corner_text)
        return plotter

    def _create_start_stop_button(self, plotter: vedo.Plotter):

        return plotter.add_button(self._start_stop_button_handler,
                                  states=["\u23F5 Play  ", "\u23F8 Pause"],
                                  font='Kanopus',
                                  size=32)

    def _start_stop_button_handler(self, object, ename):

        play = 'Play' in self._start_stop_button.status()

        if self._timer_id is not None:
            self._plotter.timer_callback('destroy', self._timer_id)
            self._timer_id = None

        if play:
            self._last_time_s = time.time()
            self._timer_id = self._plotter.timer_callback(
                'create', dt=self._params.timer_dt_ms)

        self._start_stop_button.switch()

        if self._params.hide_button_on_start:
            # REVIEW: better method?
            self._plotter.remove(self._start_stop_button)

    def show(self):

        if self._current_state == 'finished':
            return

        axes: int | dict

        if self._params.axes.custom is not None:

            axes = asdict(self._params.axes.custom)
        else:
            axes = self._params.axes.type

        self._is_show_called = True

        initial_camera = None
        if self._fixed_camera is not None:
            initial_camera = self._fixed_camera
        elif self._camera_animation is not None:
            initial_camera = self._camera_animation.cameras[0]

        self._plotter.show(viewup='z', axes=axes, camera=initial_camera)

        if self._video is not None:
            self._log.debug('closing video...')
            self._video.close()

        self._log.debug('exit')

    def _timer_handler(self, _):

        if self._current_state == 'finished':
            return

        if self._current_state == 'not_started':
            self._current_state = 'running'

        if self._last_time_s is None:
            return

        current_time_s = time.time()

        dt_s = current_time_s - self._last_time_s

        self._last_time_s = current_time_s

        assert self._current_scene_actor is not None

        frame_step_s = None
        if self._params.frame_selection == PlaybackFrameSelection.wall_time:
            assert self._params.wall_time_playback_speed is not None
            playback_speed = self._params.wall_time_playback_speed
            frame_step_s = dt_s * playback_speed
        elif self._params.frame_selection == PlaybackFrameSelection.fixed:
            assert self._params.fixed_frame_step_s is not None
            frame_step_s = self._params.fixed_frame_step_s

        assert frame_step_s is not None

        scene_status, scene_corner_text = self._current_scene_actor.step(
            dt_s=frame_step_s, plotter=self._plotter)

        if self._camera_animation is not None:

            assert self._current_scene_actor.last_time_s is not None
            scene_time_s = self._current_scene_actor.last_time_s

            if scene_time_s <= self._camera_animation.max_time_s:
                scaled_time = np.clip(scene_time_s /
                                      self._camera_animation.max_time_s,
                                      a_min=0.,
                                      a_max=1.)

                self._plotter.move_camera(
                    t=scaled_time,
                    cameras=self._camera_animation.cameras,
                    times=self._camera_animation.times,
                    smooth=self._camera_animation.smooth)

        assert self._corner_text is not None

        annotations = []
        video_speed = None
        if self._params.frame_selection == PlaybackFrameSelection.wall_time:
            video_speed = self._params.wall_time_playback_speed
        elif self._params.frame_selection == PlaybackFrameSelection.fixed and self._params.video is not None:
            assert self._params.fixed_frame_step_s is not None
            video_speed = round(self._params.fixed_frame_step_s /
                                (1 / self._params.video.fps),
                                ndigits=1)

        annotation_params = self._params.annotation
        if video_speed is not None and annotation_params.speed:
            annotations.append(f'speed: {video_speed}x')

        if annotation_params.scene:
            annotations.append(f'scene: {self._current_scene_index}')

        annotations.append(scene_corner_text)

        self._corner_text.text('; '.join(annotations))

        self._plotter.render()

        if self._params.video is not None and self._video is None:
            # create video recorder
            self._log.debug('creating video recorder: %s', self._params.video)
            self._video = vedo.Video(str(self._params.video.output_path),
                                     fps=self._params.video.fps,
                                     backend=self._params.video.backend)

        if self._video is not None:
            self._log.debug('adding video frame...')
            self._video.add_frame()

        if scene_status == 'finished':
            self._create_next_scene(start_episode=0, start_time_s=0)

    def _initialize_camera_config(self):

        if self._params.camera is not None and self._params.camera_animation is not None:
            raise ValueError(
                "Only one of 'camera' or 'camera_animation' is allowed, not both."
            )

        self._fixed_camera = None
        self._camera_animation = None

        if self._params.camera is not None:
            self._fixed_camera = asdict(self._params.camera)
        elif self._params.camera_animation is not None:
            self._camera_animation = self._create_camera_animation(
                self._params.camera_animation)

    def _create_camera_animation(
            self, params: CameraAnimationParams) -> CameraAnimation:

        keyframes = params.keyframes

        if len(keyframes) == 0:
            raise ValueError('Missing keyframes in camera_animation')

        if keyframes[0].time_s != 0.:
            raise ValueError('Missing keyframe at time 0.')

        max_time_s = keyframes[len(keyframes) - 1].time_s

        times = np.array([frame.time_s for frame in keyframes])
        times = times / max_time_s

        cameras = [asdict(key.camera) for key in keyframes]

        result = CameraAnimation(cameras=cameras,
                                 times=times.tolist(),
                                 max_time_s=max_time_s,
                                 smooth=params.smooth)
        return result

    def _create_next_scene(self, start_episode: int, start_time_s: float):

        self._destroy_current_scene()
        self._current_scene_index += 1

        finished = True
        while len(self._scenes) > self._current_scene_index:
            self._current_scene_actor, scene_state = self._create_scene(
                self._scenes[self._current_scene_index],
                start_episode=start_episode,
                start_time_s=start_time_s)

            if scene_state != 'finished':
                # found a non finished scene
                finished = False
                break

            # try the next
            self._destroy_current_scene()
            self._current_scene_index += 1

        if finished:
            self._current_state = 'finished'
            self._destroy_callbacks()

            if self._is_show_called:
                self._plotter.close()

    def _create_scene(self, scene: Scene, start_episode: int,
                      start_time_s: float):

        scene_actor = AirVelocitySceneActor(scene=scene,
                                            params=self._params,
                                            start_episode=start_episode,
                                            start_time_s=start_time_s)
        scene_state = scene_actor.create(self._plotter)
        return scene_actor, scene_state

    def _destroy_current_scene(self):
        if self._current_scene_actor is None:
            return

        self._current_scene_actor.destroy(self._plotter)
        self._current_scene_actor = None
