from typing import Literal
from dataclasses import dataclass
from pathlib import Path
import functools
from ..common.dataset_utils import save_figure
from thermal.visualization.thermal_cross_section_plot import VerticalPlotParameters
from thermal.visualization import ThermalCrossSectionPlot, ThermalPlotDataSerializer
from utils.plot_style import select_plot_style
from .load_close_distance_dataset import load_close_distance_dataset
import tyro
from utils.logging import configure_logger
import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from bird_comparison.analysis.common.thermal_map import resolve_thermal_mapping


@dataclass
class DataSourceParams:
    policy_neptune_run_id: str
    episode_count: int


@dataclass
class ThermalTrajectoryPlotParams:
    thermal: str
    episode: int
    bird_name: str
    setup: Literal['student_alone', 'student_with_birds']


@dataclass
class Config:
    datasource: DataSourceParams
    plot: ThermalTrajectoryPlotParams


def main(
    config: Config,
    output_dir: Path = Path(
        'results/analysis/from_close_distance_using_bird_wing_loadings/trajectory_plot'
    )):

    datasource = config.datasource
    plot_params = config.plot

    configure_logger()
    ds = load_close_distance_dataset(
        policy_neptune_run_id=datasource.policy_neptune_run_id,
        episode_count=datasource.episode_count)

    # select thermal, bird, episode

    thermal_id = plot_params.thermal
    thermal = resolve_thermal_mapping()[thermal_id]
    episode = plot_params.episode
    bird_name = plot_params.bird_name
    setup = plot_params.setup

    # select ai trajectory
    ai_ds = ds.sel(setup=setup,
                   thermal=thermal,
                   episode=episode,
                   bird_name=bird_name)
    ai_ds = ai_ds[[
        'position_earth_m_x', 'position_earth_m_y', 'position_earth_m_z'
    ]]
    ai_ds = ai_ds.dropna(dim='time_s')

    bird_ds = ds.sel(setup='birds',
                     thermal=thermal,
                     episode=0,
                     bird_name=bird_name)
    bird_ds = bird_ds[[
        'position_earth_m_x', 'position_earth_m_y', 'position_earth_m_z'
    ]]
    bird_ds = bird_ds.dropna(dim='time_s')

    ai_x = ai_ds['position_earth_m_x']
    ai_y = ai_ds['position_earth_m_y']
    ai_z = ai_ds['position_earth_m_z']

    bird_x = bird_ds['position_earth_m_x']
    bird_y = bird_ds['position_earth_m_y']
    bird_z = bird_ds['position_earth_m_z']

    thermal_id = ai_ds['thermal_id'].item()

    output_name = f'{thermal_id}-{bird_name}-{episode}'

    _, vertical_plot_data = _load_thermal_plot_data(thermal_id=thermal_id)

    # create the figure
    _ADDITIONAL_RC_PARAMS = {
        'savefig.bbox': None,  # the default 'tight' caused colorbar issues
    }

    # using Agg engine, because with Cairo, there are positional issues at the axes
    with select_plot_style('science',
                           _ADDITIONAL_RC_PARAMS,
                           engine_override='Agg'):

        cm_to_inch = 1 / 2.54

        axis_width = 8.5 * cm_to_inch
        axis_height = 6.0 * cm_to_inch
        main_left_right_padding = 1.2 * cm_to_inch
        main_top_padding = 0.1 * cm_to_inch
        main_bottom_padding = 0.8 * cm_to_inch

        colorbar_width = 0.4 * cm_to_inch
        colorbar_left_padding = 0.2 * cm_to_inch
        colorbar_vertical_padding = 1.3 * cm_to_inch

        horizontal = [
            Size.Fixed(main_left_right_padding),
            Size.Fixed(axis_width),
            Size.Fixed(colorbar_left_padding),
            Size.Fixed(colorbar_width),
            Size.Fixed(main_left_right_padding)
        ]
        vertical = [
            Size.Fixed(main_bottom_padding),
            Size.Fixed(colorbar_vertical_padding),
            Size.Fixed(axis_height - 2 * colorbar_vertical_padding),
            Size.Fixed(colorbar_vertical_padding),
            Size.Fixed(main_top_padding),
        ]

        horizontal_figsize = functools.reduce(lambda x, y: x + y.fixed_size,
                                              horizontal, 0.)
        vertical_figsize = functools.reduce(lambda x, y: x + y.fixed_size,
                                            vertical, 0.)

        fig = plt.figure(figsize=(horizontal_figsize, vertical_figsize))
        divider = Divider(fig,
                          pos=(0, 0, 1, 1),
                          horizontal=horizontal,
                          vertical=vertical,
                          aspect=False)

        ax = fig.add_axes(divider.get_position(),
                          axes_locator=divider.new_locator(nx=1, ny=1, ny1=4))
        colorbar_ax = fig.add_axes(divider.get_position(),
                                   axes_locator=divider.new_locator(nx=3,
                                                                    nx1=4,
                                                                    ny=2,
                                                                    ny1=3))

        cross_section_plot = ThermalCrossSectionPlot()

        cross_section_plot.plot_vertical_cross_section(
            ax=ax,
            data=vertical_plot_data,
            params=VerticalPlotParameters(projection='YZ',
                                          projection_type='max'),
            x_lim=(0., 600.),
            y_lim=(850., 1270.),
            show_w_min_max=False,
            show_title=False)

        ax.set_aspect('equal', adjustable='datalim')

        cross_section_plot.place_colorbar(fig,
                                          colorbar_ax=colorbar_ax,
                                          ticks=1.0)

        ax.plot(ai_y, ai_z, label='AI', color='#0c5da5ff', linewidth=0.5)
        ax.plot(bird_y,
                bird_z,
                label='Bird',
                color='white',
                linewidth=0.5,
                zorder=1)
        ax.plot(bird_y,
                bird_z,
                label='Bird',
                color='#00d60aff',
                linestyle='dashed',
                linewidth=0.5,
                zorder=2)

        inset_xlim = (10, 130)
        inset_ylim = (900, 1000)

        inset_ax = zoomed_inset_axes(ax, zoom=2, loc='lower right')
        inset_ax.plot(ai_y, ai_z, label='AI', color='#0c5da5ff', linewidth=1.)
        inset_ax.plot(bird_y,
                      bird_z,
                      label='Bird',
                      color='white',
                      linewidth=1.)
        inset_ax.plot(bird_y,
                      bird_z,
                      label='Bird',
                      color='#00d60aff',
                      linestyle='dashed',
                      linewidth=1.)
        inset_ax.set_xlim(*inset_xlim)
        inset_ax.set_ylim(*inset_ylim)

        cross_section_plot.plot_vertical_cross_section(
            ax=inset_ax,
            data=vertical_plot_data,
            params=VerticalPlotParameters(projection='YZ',
                                          projection_type='max'),
            x_lim=inset_xlim,
            y_lim=inset_ylim,
            show_w_min_max=False,
            show_title=False)

        _clear_inset_axes(inset_ax)
        mark_inset(ax, inset_ax, loc1=2, loc2=3, fc="none", ec="0.5")

        ax.legend()

        save_figure(fig, output_dir=output_dir, name=output_name)

        plt.show()


def _clear_inset_axes(inset_ax: matplotlib.axes.Axes):
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel('')
    inset_ax.set_ylabel('')
    inset_ax.set_title('')


def _load_thermal_plot_data(thermal_id: str):
    serializer = ThermalPlotDataSerializer()

    thermal_plot_path = Path(
        'data/bird_comparison/processed/thermal_plots/realistic'
    ) / thermal_id / f'{thermal_id}.hdf5'

    horizontal, vertical = serializer.load(str(thermal_plot_path))

    return horizontal, vertical


if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
