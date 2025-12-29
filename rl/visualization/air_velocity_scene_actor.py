from typing import Literal, Any, cast
import logging
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from dataclasses import dataclass, asdict, replace
import json
import hashlib
import h5py
import numpy as np
from utils.vector import VectorNx3
import vedo
import matplotlib as mpl
from omegaconf import MISSING
import hydra

from utils import Vector3, RandomGeneratorState
from thermal.api import AirVelocityFieldConfigBase, AirVelocityFieldInterface
from env.glider.base import SimulationBoxParameters
from vedo.vtkclasses import vtkDoubleArray, vtkScalarBarActor
from .calculate_resolution import calculate_resolution

from .glider_agent_actor import (AgentEpisode, AgentDisplayParams,
                                 FlagpostParams, GliderAgentActor, MeshParams,
                                 TrailParams, VelocityArrowParams)
from .air_velocity_coordinate_sampler import AirVelocitySamplerConfigBase


@dataclass
class ColorMap:
    cmap: np.ndarray
    vmin: float
    vmax: float


@dataclass
class ThresholdParams:
    above: float
    below: float
    scalar: str


@dataclass
class PointFilterParams:
    max_points: int | None
    threshold: ThresholdParams | None = None


@dataclass
class StreamlineParams:
    max_propagation_m: float
    initial_step_size: float
    integrator: str
    cmap_scalar: str
    alpha: float
    linewidth: float
    seed_filter: PointFilterParams | None


@dataclass
class ArrowParams:
    scale: float
    alpha: float
    color: str


@dataclass
class PointParams:
    r_m: float
    cmap_scalar: str
    filter: PointFilterParams | None


@dataclass
class AlphaFixedPointParams:
    value: float
    opacity: float


@dataclass
class VolumeAlphaParams:
    curve: list[AlphaFixedPointParams]
    attenuation_unit_length_m: float


@dataclass
class VolumeDisplayParams:
    scalar: str
    alpha: VolumeAlphaParams
    render_mode: str  # composite | max_projection


@dataclass
class ColorMapParameters:
    min_value: float
    max_value: float


@dataclass
class AirVelocityDisplayParams:
    volume: VolumeDisplayParams
    streamlines: StreamlineParams | None
    arrows: ArrowParams | None
    points: PointParams | None
    thermal_color_map: ColorMapParameters


@dataclass
class AirVelocityVolumeParams:
    spacing_m: Any  # float | Vector3


@dataclass
class AirVelocityParams:
    field: AirVelocityFieldConfigBase = MISSING
    sampler: AirVelocitySamplerConfigBase = MISSING
    volume: AirVelocityVolumeParams = MISSING
    display: AirVelocityDisplayParams = MISSING


@dataclass
class CacheParams:
    binary_volume_cache: bool = False  # does not decrease the file size
    use_cached: bool = True
    key: str | None = None


@dataclass
class AgentInstanceParams:
    color_index: int
    flagpost: FlagpostParams | None
    velocity_arrow: bool
    mesh: MeshParams


@dataclass
class AgentsParams:
    trail: TrailParams | None
    colormap_name: str
    map: dict[str, AgentInstanceParams]
    velocity_arrow: VelocityArrowParams


@dataclass
class LatexTextParams:
    text: str
    pos: list[float]
    size: float


@dataclass(kw_only=True)
class ColorbarParams:
    title: str | None
    label_format: str
    pos: list[float]
    values: list[float] | None = None
    latex: LatexTextParams | None


@dataclass(kw_only=True)
class AnnotationParams:
    speed: bool
    scene: bool
    episode: bool
    time: bool


@dataclass(kw_only=True)
class SceneParams:
    cache: CacheParams
    air_velocity: AirVelocityParams
    agents: AgentsParams
    minimum_display_box: SimulationBoxParameters
    colorbar: ColorbarParams | None
    annotation: AnnotationParams
    world_box: bool


@dataclass
class ThermalParams:
    thermal_name: str
    thermal_params: str
    thermal_random_state: RandomGeneratorState | None


@dataclass
class Episode:
    agents: list[AgentEpisode]


@dataclass
class Scene:
    thermal: ThermalParams
    episodes: list[Episode]


@dataclass
class AirVelocityFieldData:
    xyz: VectorNx3
    uvw: VectorNx3


@dataclass
class AirVelocityVolumeData:
    volume: vedo.Volume
    points: vedo.Points


SceneState = Literal['not_started'] | Literal['running'] | Literal['finished']


class AirVelocitySceneActor:

    _log: logging.Logger

    _scene: Scene
    _params: SceneParams

    _current_objects: list[vedo.CommonVisual]
    _timer_text: vedo.CornerAnnotation | None

    _current_state: SceneState
    _last_time_s: float | None

    _current_episode_index: int

    _current_agents: list[GliderAgentActor]
    _agent_color_map: dict[str, np.ndarray]

    def __init__(self, scene: Scene, params: SceneParams, start_episode: int,
                 start_time_s: float):

        self._log = logging.getLogger(__class__.__name__)

        self._scene = scene
        self._params = params

        self._current_objects = []

        self._current_state = 'not_started'
        self._last_time_s = None

        if start_episode >= len(scene.episodes):
            raise ValueError(
                f'invalid start_episode index {start_episode}; episode count: {len(scene.episodes)}'
            )

        self._current_episode_index = start_episode - 1
        self._current_agents = []

        self._agent_color_map = self._create_agent_name_color_mapping(
            self._params.agents.colormap_name, {
                k: v.color_index
                for k, v in self._params.agents.map.items()
            })

        self._select_next_episode(start_time_s=start_time_s)

    @property
    def last_time_s(self):
        return self._last_time_s

    def create(self, plotter: vedo.Plotter):

        params = self._params

        cache_params = params.cache
        air_velocity_params = params.air_velocity

        # REVIEW: we inject random_state here
        field_config = self._prepare_air_velocity_field_config(
            air_velocity_params.field)
        air_velocity_params = replace(air_velocity_params, field=field_config)

        cache_dir = self._cache_dir(cache_key=cache_params.key,
                                    air_velocity_params=air_velocity_params)
        can_use_cache = cache_params.use_cached

        # create or use cached air velocity field
        air_velocity_data, data_from_cache = self._use_cached_or_create_air_velocity_field(
            params=air_velocity_params,
            cache_dir=cache_dir,
            can_use_cache=can_use_cache)

        # create or use cached volume
        volume_data = self._use_cached_or_interpolate_air_velocity_volume(
            air_velocity_data=air_velocity_data,
            volume_params=air_velocity_params.volume,
            cache_dir=cache_dir,
            can_use_cache=data_from_cache,
            binary_cache=cache_params.binary_volume_cache)

        # configure the volume
        objects_to_show = self._configure_air_velocity_field_volume(
            data=volume_data, params=air_velocity_params.display)

        self._timer_text = vedo.CornerAnnotation()

        objects_to_show = [*objects_to_show, self._timer_text]

        world = self._create_world(box=params.minimum_display_box,
                                   air_velocity_volume=volume_data.volume)
        if self._params.world_box:
            objects_to_show.append(world)

        plotter.add(*objects_to_show)
        self._current_objects = objects_to_show

        state, _ = self.step(dt_s=0., plotter=plotter)
        return state

    def _create_world(self, box: SimulationBoxParameters,
                      air_velocity_volume: vedo.CommonAlgorithms):

        x_min, x_max, y_min, y_max, z_min, z_max = air_velocity_volume.bounds()
        assert box.limit_earth_xyz_low_m is not None and box.limit_earth_xyz_high_m is not None
        pos = [
            min(x_min, box.limit_earth_xyz_low_m[0]),
            max(x_max, box.limit_earth_xyz_high_m[0]),
            min(y_min, box.limit_earth_xyz_low_m[1]),
            max(y_max, box.limit_earth_xyz_high_m[1]),
            min(z_min, box.limit_earth_xyz_low_m[2]),
            max(z_max, box.limit_earth_xyz_high_m[2])
        ]
        return vedo.Box(pos=pos).wireframe()

    def _prepare_air_velocity_field_config(self,
                                           config: AirVelocityFieldConfigBase):
        if self._scene.thermal.thermal_random_state is not None:
            self._log.debug(
                'injecting random_state to air velocity config: %s',
                self._scene.thermal.thermal_random_state)
            # REVIEW: some method to pass random state in other way
            config = cast(
                AirVelocityFieldConfigBase, {
                    **asdict(config), "random_state":
                    self._scene.thermal.thermal_random_state
                })

        return config

    def _use_cached_or_create_air_velocity_field(self,
                                                 params: AirVelocityParams,
                                                 cache_dir: Path,
                                                 can_use_cache: bool):

        air_velocity_cache_file = cache_dir / 'air_velocity_field.h5'
        cache_meta_filepath = cache_dir / 'air_velocity_field.json'

        cache_meta = self._air_velocity_cache_meta(params)

        # check meta
        if cache_meta_filepath.exists():
            loaded_cache_meta = self._load_cache_meta(cache_meta_filepath)

            if loaded_cache_meta.strip() != cache_meta.strip():
                self._log.debug(
                    'params changed for air velocity field, cannot use cached volume'
                )
                can_use_cache = False
        else:
            can_use_cache = False

        if can_use_cache:
            self._log.debug('checking air velocity field cache %s',
                            air_velocity_cache_file)

            if air_velocity_cache_file.exists():
                self._log.debug('using cached air velocity field: %s',
                                air_velocity_cache_file)
                return self._load_air_velocity_cache(
                    air_velocity_cache_file), True

            self._log.debug('cache not found, querying air velocity field...')
        else:
            self._log.debug('cannot use cache, querying air velocity field...')

        air_velocity_data = self._create_and_query_air_velocity_field(
            air_velocity_config=params.field, sampler_config=params.sampler)

        self._log.debug('saving air velocity field cache to %s...',
                        air_velocity_cache_file)
        self._save_air_velocity_cache(air_velocity_cache_file,
                                      air_velocity_data)
        self._save_cache_meta(cache_meta_filepath, cache_meta)

        return air_velocity_data, False

    def _save_cache_meta(self, filepath: Path, meta: str):
        with open(filepath, 'w') as file:
            file.write(meta)

    def _load_cache_meta(self, filepath: Path) -> str:
        with open(filepath, 'r') as file:
            return file.read()

    def _generate_cache_key(self, params: AirVelocityParams):

        # create the cache key
        cache_json = self._air_velocity_cache_meta(params)
        cache_key = hashlib.md5(cache_json.encode()).hexdigest()

        return cache_key

    def _air_velocity_cache_meta(self, params: AirVelocityParams):
        params_dict = asdict(params)

        # use only the field and the sampler
        cache_dict = {}
        cache_dict['field'] = params_dict['field']
        cache_dict['sampler'] = params_dict['sampler']

        # create the cache key
        meta_json = json.dumps(cache_dict, sort_keys=True)

        return meta_json

    def _air_velocity_volume_cache_meta(self, params: AirVelocityVolumeParams):
        cache_dict = asdict(params)

        # create the cache key
        meta_json = json.dumps(cache_dict, sort_keys=True)

        return meta_json

    def _cache_dir(self, cache_key: str | None,
                   air_velocity_params: AirVelocityParams):

        if cache_key is None:
            cache_key = self._generate_cache_key(air_velocity_params)

        base_dir = Path(__file__).resolve().parents[2]
        cache_dir = base_dir / 'data' / 'cache' / 'player' / cache_key

        return cache_dir

    def _load_air_velocity_cache(self, filepath: Path) -> AirVelocityFieldData:

        with h5py.File(filepath, 'r') as f:
            xyz = self._read_dataset_as_nparray(f, 'xyz')
            uvw = self._read_dataset_as_nparray(f, 'uvw')

            return AirVelocityFieldData(xyz=xyz, uvw=uvw)

    def _save_air_velocity_cache(self, filepath: Path,
                                 data: AirVelocityFieldData):

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            compression = 'gzip'
            f.create_dataset('xyz', data=data.xyz, compression=compression)
            f.create_dataset('uvw', data=data.uvw, compression=compression)

    def _read_dataset_as_nparray(self, parent: h5py.Group, key: str):
        dataset = parent[key]
        assert isinstance(dataset, h5py.Dataset)

        return np.array(dataset)

    def _create_and_query_air_velocity_field(
            self, air_velocity_config: AirVelocityFieldConfigBase,
            sampler_config: AirVelocitySamplerConfigBase
    ) -> AirVelocityFieldData:

        self._log.debug('creating coordinates to query...')

        air_velocity_field = self._create_air_velocity_field(
            config=air_velocity_config)
        coordinate_sampler = hydra.utils.instantiate(sampler_config)

        x, y, z = coordinate_sampler.sample(
            air_velocity_field=air_velocity_field)

        # get the air velocities
        self._log.debug(
            'querying air velocity field; x.shape=%s,y.shape=%s,z.shape=%s',
            x.shape, y.shape, z.shape)
        uvw, _ = air_velocity_field.get_velocity(x_earth_m=x,
                                                 y_earth_m=y,
                                                 z_earth_m=z,
                                                 t_s=0)

        assert uvw.shape[0] == 3
        assert uvw.ndim == 2
        # [3, N] -> [N, 3]
        uvw = uvw.T

        xyz = np.stack((x, y, z), axis=1)

        return AirVelocityFieldData(xyz=xyz, uvw=uvw)

    def destroy(self, plotter: vedo.Plotter):
        plotter.remove(self._current_objects)
        self._current_objects = []

    def step(self, dt_s: float,
             plotter: vedo.Plotter) -> tuple[SceneState, str]:

        assert self._current_state != 'not_started', 'The actor should have been started'

        if self._current_state == 'finished':
            self._log.debug('scene is finished')
            return self._current_state, 'finished'

        if self._last_time_s is None:
            current_time_s = 0.
        else:
            current_time_s = self._last_time_s + dt_s

        self._log.debug('step; current_time_s=%f;dt_s=%f', current_time_s,
                        dt_s)

        assert len(self._current_agents) > 0, 'there is no current agent'

        all_agents_finished = True
        for agent in self._current_agents:
            agent_state = agent.step(current_time_s=current_time_s,
                                     plotter=plotter)
            if agent_state != 'finished':
                all_agents_finished = False

        self._last_time_s = current_time_s

        if all_agents_finished:
            # create the next agent
            self._select_next_episode(start_time_s=0)

        if self._current_state == 'finished':
            self._log.debug('scene is finished')
            return self._current_state, 'finished'

        annotation_params = self._params.annotation
        annotations = []
        if annotation_params.episode:
            annotations.append(f'episode: {self._current_episode_index}')

        if annotation_params.time:
            annotations.append(f'time: {current_time_s:.2f}s')

        annotation = '; '.join(annotations)

        return self._current_state, annotation

    def _select_next_episode(self, start_time_s: float):
        self._current_episode_index += 1
        if len(self._scene.episodes) > self._current_episode_index:
            self._log.debug('playing episode %d', self._current_episode_index)

            episode = self._scene.episodes[self._current_episode_index]
            self._current_agents = self._create_agent_actors(episode)
            self._current_state = 'running'
            if start_time_s > 0:
                self._last_time_s = start_time_s
            else:
                self._last_time_s = None

        else:
            # no more episodes
            self._log.debug('no more episodes, finish the scene')
            self._current_state = 'finished'

    def _create_agent_actors(self, episode: Episode) -> list[GliderAgentActor]:

        agent_actors = []

        default_params = AgentInstanceParams(
            color_index=0,
            flagpost=FlagpostParams(show_agent_name=True),
            velocity_arrow=True,
            mesh=MeshParams(path=Path('glider.ply'), scale=10.))

        for agent_episode in episode.agents:

            agent_color = None
            agent_name = agent_episode.agent.agent_name

            if agent_name in self._agent_color_map:
                agent_color = self._agent_color_map[agent_name]

            if agent_name in self._params.agents.map:
                agent_params = self._params.agents.map[agent_name]
            else:
                agent_params = default_params

            velocity_arrow_params = (self._params.agents.velocity_arrow
                                     if agent_params.velocity_arrow == True
                                     else None)

            agent_display_params = AgentDisplayParams(
                color=agent_color,
                trail=self._params.agents.trail,
                flagpost=agent_params.flagpost,
                velocity_arrow=velocity_arrow_params,
                mesh=agent_params.mesh)

            agent_actor = GliderAgentActor(agent_episode,
                                           display_params=agent_display_params)
            agent_actors.append(agent_actor)

        return agent_actors

    def _create_agent_name_color_mapping(
            self, colormap_name: str,
            agent_color_index_map: dict[str, int]) -> dict[str, np.ndarray]:

        colormap = mpl.colormaps[colormap_name]

        if isinstance(colormap, LinearSegmentedColormap):
            max_color_index = np.max(list(agent_color_index_map.values()))
            index_color_map = self._generate_discrete_colormap_from_linear(
                colormap, max_color_index + 1)
        elif isinstance(colormap, ListedColormap):
            index_color_map = [np.array(c) for c in colormap.colors]
        else:
            assert False, f'unsupported color map: {colormap}'

        result = {
            agent_id: index_color_map[color_index % len(index_color_map)]
            for agent_id, color_index in agent_color_index_map.items()
        }

        return result

    def _generate_discrete_colormap_from_linear(
            self, colormap: LinearSegmentedColormap,
            max_colors: int) -> list[np.ndarray]:

        colors = []
        for i in range(max_colors):
            color = vedo.color_map(i, colormap, vmin=0, vmax=max_colors - 1)
            colors.append(color)

        return colors

    def _use_cached_or_interpolate_air_velocity_volume(
            self, air_velocity_data: AirVelocityFieldData,
            volume_params: AirVelocityVolumeParams, cache_dir: Path,
            can_use_cache: bool, binary_cache: bool):

        cache_meta = self._air_velocity_volume_cache_meta(volume_params)

        # check meta
        cache_meta_filepath = self._volume_cache_meta_filename(cache_dir)
        if cache_meta_filepath.exists():
            loaded_cache_meta = self._load_cache_meta(cache_meta_filepath)

            if loaded_cache_meta.strip() != cache_meta.strip():
                self._log.debug(
                    'params changed for volume, cannot use cached volume')
                can_use_cache = False
        else:
            can_use_cache = False

        if can_use_cache:
            volume_data = self._try_load_air_velocity_volume_cache(
                cache_dir=cache_dir)

            if volume_data is not None:
                self._log.debug('using cached air velocity volume')
                return volume_data
            else:
                self._log.debug(
                    'air velocity volume cache not found, recreating...')
        else:
            self._log.debug('cannot use cached volume, recreating...')

        volume_data = self._create_air_velocity_field_volume(
            air_velocity_data, params=volume_params)

        self._save_air_velocity_volume_cache(cache_dir=cache_dir,
                                             data=volume_data,
                                             binary=binary_cache)
        self._save_cache_meta(cache_meta_filepath, cache_meta)

        return volume_data

    def _try_load_air_velocity_volume_cache(
            self, cache_dir: Path) -> AirVelocityVolumeData | None:

        # load the volume
        volume_filepath = self._volume_cache_filename(cache_dir)
        if not volume_filepath.exists():
            return None
        volume = vedo.Volume(str(volume_filepath))

        # load the point cloud
        points_filepath = self._volume_points_cache_filename(cache_dir)
        if not points_filepath.exists():
            return None

        points = vedo.Points(str(points_filepath))

        return AirVelocityVolumeData(volume=volume, points=points)

    def _save_air_velocity_volume_cache(self, cache_dir: Path,
                                        data: AirVelocityVolumeData,
                                        binary: bool):

        cache_dir.mkdir(parents=True, exist_ok=True)

        # save the volume
        volume_filepath = self._volume_cache_filename(cache_dir)
        data.volume.write(str(volume_filepath), binary=binary)

        # save the point cloud
        points_filepath = self._volume_points_cache_filename(cache_dir)
        data.points.write(str(points_filepath), binary=binary)

    def _volume_cache_meta_filename(self, cache_dir: Path):
        return cache_dir / 'air_velocity_field_volume.json'

    def _volume_cache_filename(self, cache_dir: Path):
        return cache_dir / 'air_velocity_field_volume.vti'

    def _volume_points_cache_filename(self, cache_dir: Path):
        return cache_dir / 'air_velocity_field_points.vtp'

    def _create_air_velocity_field_volume(
            self, air_velocity_data: AirVelocityFieldData,
            params: AirVelocityVolumeParams) -> AirVelocityVolumeData:

        source_points = vedo.Points(air_velocity_data.xyz)
        source_points.pointdata['u'] = air_velocity_data.uvw[:, 0]
        source_points.pointdata['v'] = air_velocity_data.uvw[:, 1]
        source_points.pointdata['w'] = air_velocity_data.uvw[:, 2]

        # calculate the resolution
        x_min, x_max, y_min, y_max, z_min, z_max = source_points.bounds()
        size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])

        spacing_m = params.spacing_m
        spacing_m_arr = np.asarray(spacing_m)
        assert np.isscalar(spacing_m) or spacing_m_arr.size == 3
        resolution = calculate_resolution(box_size=size,
                                          spacing_m=spacing_m_arr)
        assert isinstance(resolution, np.ndarray)
        assert resolution.size == 3

        # create the volume
        dims = list(resolution)

        self._log.debug(
            'interpolating to create volume using resolution %s...', dims)
        volume = source_points.tovolume(kernel='gaussian', n=5, dims=dims)

        self._log.debug('air velocity field volume created')

        return AirVelocityVolumeData(volume=volume, points=source_points)

    def _configure_air_velocity_field_volume(
            self, data: AirVelocityVolumeData,
            params: AirVelocityDisplayParams) -> list[vedo.CommonVisual]:

        volume = data.volume
        points = data.points
        original_points = data.points

        thermal_color_map = self._create_thermal_colormap(
            params.thermal_color_map)

        objects_to_show: list[vedo.CommonVisual] = []
        volume, volume_widgets = self._configure_volume(
            volume=volume, params=params.volume, color_map=thermal_color_map)
        objects_to_show.append(volume)
        objects_to_show.append(*volume_widgets)

        if params.points:
            points = self._configure_points(points=points,
                                            params=params.points,
                                            color_map=thermal_color_map)

            objects_to_show.append(points)

        if params.streamlines:

            streamlines = self._create_streamlines(volume=volume,
                                                   seeds=original_points,
                                                   params=params.streamlines,
                                                   color_map=thermal_color_map)
            if streamlines is not None:
                objects_to_show.append(streamlines)

        if params.arrows:
            arrows = self._create_arrows(points=points, params=params.arrows)
            objects_to_show.append(arrows)

        return objects_to_show

    def _configure_volume(self, volume: vedo.Volume,
                          params: VolumeDisplayParams, color_map: ColorMap):

        # select specific scalar for the coloring
        volume.pointdata.select(params.scalar)

        # configure alpha
        alpha_list = [(point.value, point.opacity)
                      for point in params.alpha.curve]
        volume.alpha(alpha_list)
        volume.alpha_unit(params.alpha.attenuation_unit_length_m)

        # rendering mode
        render_mode = None
        if params.render_mode == 'composite':
            render_mode = 0
        elif params.render_mode == 'max_projection':
            render_mode = 1
        else:
            raise Exception(f'invalid render mode: {params.render_mode}')

        volume.mode(render_mode)

        # set the color map
        volume.color(color_map.cmap, vmin=color_map.vmin, vmax=color_map.vmax)

        widgets = []

        if self._params.colorbar is not None:

            colorbar_params = self._params.colorbar

            volume.add_scalarbar(label_format=colorbar_params.label_format,
                                 pos=colorbar_params.pos,
                                 title='' if colorbar_params.title is None else
                                 colorbar_params.title)

            scalarbar = cast(vtkScalarBarActor, volume.scalarbar)

            if colorbar_params.values is not None:
                label_values = vtkDoubleArray()
                label_values.SetNumberOfValues(len(colorbar_params.values))
                for i in range(len(colorbar_params.values)):
                    label_values.SetValue(i, colorbar_params.values[i])

                scalarbar.SetCustomLabels(label_values)
                scalarbar.SetUseCustomLabels(True)

            if colorbar_params.latex is not None:
                latex_params = colorbar_params.latex
                latex_label = vedo.Latex(latex_params.text,
                                         s=1.).clone2d(pos=latex_params.pos,
                                                       size=latex_params.size)
                widgets.append(latex_label)

        return volume, widgets

    def _configure_points(self, points: vedo.Points, params: PointParams,
                          color_map: ColorMap) -> vedo.Points:

        if params.filter is not None:
            points = self._filter_points(points, params=params.filter)

        points.cmap(color_map.cmap,
                    params.cmap_scalar,
                    vmin=color_map.vmin,
                    vmax=color_map.vmax)
        points.point_size(params.r_m)

        return points

    def _filter_points(self, points: vedo.Points, params: PointFilterParams):
        if params.threshold:
            points = self._threshold_points(points, params=params.threshold)

        if params.max_points is not None:
            points = self._filter_points_based_on_uniform_indices(
                points, params.max_points)

        return points

    def _threshold_points(self, points: vedo.Points, params: ThresholdParams):
        points = points.clone()
        points = points.threshold(above=params.above,
                                  below=params.below,
                                  scalars=params.scalar)
        return points

    def _filter_points_based_on_uniform_indices(self, points: vedo.Points,
                                                n: int):

        if points.nvertices <= n:
            return points

        # uniform randomly select n indices and filter the points
        indices = np.random.choice(points.nvertices, size=n, replace=False)

        filtered_points = vedo.Points(points.vertices[indices])
        for key in points.pointdata.keys():
            arr = points.pointdata[key]
            assert isinstance(arr, np.ndarray)
            filtered_points.pointdata[key] = arr[indices]

        return filtered_points

    def _create_arrows(self, points: vedo.Points,
                       params: ArrowParams) -> vedo.Arrows:

        points_u = np.array(points.pointdata['u'])
        points_v = np.array(points.pointdata['v'])
        points_w = np.array(points.pointdata['w'])
        points_uvw = np.stack((points_u, points_v, points_w), axis=1)

        points_coords = points.vertices
        arrows = vedo.Arrows(start_pts=points_coords,
                             end_pts=points_coords + points_uvw * params.scale,
                             c=params.color
                             # thickness=0.5,
                             )

        arrows.alpha(params.alpha)

        return arrows

    def _create_streamlines(self, volume: vedo.Volume, seeds: vedo.Points,
                            params: StreamlineParams, color_map: ColorMap):

        if params.seed_filter is not None:
            seeds = self._filter_points(seeds, params.seed_filter)

        volume_u = np.array(volume.pointdata['u'])
        volume_v = np.array(volume.pointdata['v'])
        volume_w = np.array(volume.pointdata['w'])
        volume.pointdata['uvw'] = np.stack((volume_u, volume_v, volume_w),
                                           axis=1)

        streamlines = volume.compute_streamlines(
            seeds=seeds,
            max_propagation=params.max_propagation_m,
            initial_step_size=params.initial_step_size,
            integrator=params.integrator,
            # step_length=0.1,
            compute_vorticity=False,
            surface_constrained=False)

        if streamlines is not None:

            # sample scalar values from the volume at the streamline points
            streamline_points = vedo.Points(streamlines.vertices).probe(volume)

            streamlines.alpha(params.alpha)
            streamlines.linewidth(params.linewidth)

            # set the color map
            streamline_color_source = np.array(
                streamline_points.pointdata[params.cmap_scalar])
            streamlines.cmap(color_map.cmap,
                             streamline_color_source,
                             on='points',
                             vmin=color_map.vmin,
                             vmax=color_map.vmax)

        return streamlines

    def _create_thermal_colormap(self, params: ColorMapParameters):
        colors = ['#a34d93', '#c4deff', '#ffff3d', '#ff3333', '#9c0000']

        cmap_whole = params.max_value - params.min_value
        zero_pos = np.abs(params.min_value) / cmap_whole

        nodes = [0.0, zero_pos, zero_pos + (1 - zero_pos) / 3, 0.8, 1.0]
        cmap = list(zip(nodes, colors))

        linear_segment_map = LinearSegmentedColormap.from_list(
            'thermal_cmap', cmap)

        vedo_cmap = vedo.color_map(range(256), linear_segment_map)

        result = ColorMap(cmap=vedo_cmap,
                          vmin=params.min_value,
                          vmax=params.max_value)
        return result

    def _calculate_resolution(self, box_size: Vector3 | float,
                              spacing_m: float) -> Vector3 | int:
        result = (np.asarray(box_size, dtype=float) / spacing_m +
                  1).astype(int)
        if np.isscalar(box_size):
            return int(result)
        else:
            return result

    def _create_air_velocity_field(
            self,
            config: AirVelocityFieldConfigBase) -> AirVelocityFieldInterface:

        self._log.debug('creating air velocity field...')

        air_velocity_field: AirVelocityFieldInterface = hydra.utils.instantiate(
            config, _convert_='object')

        self._log.debug('air velocity field created; air_velocity_field=%s',
                        air_velocity_field)

        air_velocity_field.reset()

        return air_velocity_field
