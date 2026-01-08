from pathlib import Path
from itertools import chain
import matplotlib
from matplotlib.colors import ListedColormap
from utils.logging import configure_logger
from utils.plot_style import select_plot_style
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.axes
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
import tyro
import seaborn as sns

from .load_per_distance_dataset import load_per_distance_dataset
from ..common.dataset_utils import (save_figure, load_bird_dataset,
                                    write_dataframe)
from ..common.add_altitude_achievement_percent import (
    add_agent_initial_and_maximum_altitude, add_altitude_achievement_percent)


def generate_wind_speed_analysis(
    ds: xr.Dataset,
    output_dir: Path = Path(
        'results/analysis/from_different_distances_using_train_glider/wind_speed'
    )):

    # preprocess
    ds = add_agent_initial_and_maximum_altitude(ds)
    ds = add_altitude_achievement_percent(
        ds,
        bird_maximum_altitude_reference_variable='thermal_bird_maximum_altitude'
    )

    _plot_wind_speed_with_achievement_percent(ds, output_dir)
    _plot_thermal_vertical_velocity(ds, output_dir)
    _plot_altitude_achievement_percent_for_different_wind_speeds_scatter(
        ds, output_dir)
    _plot_wind_speed(ds, output_dir)
    _plot_thermal_bird_vertical_velocity(output_dir)


def _plot_altitude_achievement_percent_for_different_wind_speeds_scatter(
        ds: xr.Dataset, output_dir: Path):

    _ADDITIONAL_RC_PARAMS = {}

    thermal_mean_wind_speed_da = ds['horizontal_air_velocity_earth_m_s'].mean(
        dim=['setup', 'start_distance', 'episode', 'time_s'])

    ds = ds.assign(thermal_mean_wind_speed=thermal_mean_wind_speed_da)
    ds = ds.assign(
        thermal_altitude_achievement_percent=ds['altitude_achievement_percent']
        .mean(dim=['episode']))

    distance_mean_ds = xr.Dataset({
        'thermal_mean_wind_speed':
        thermal_mean_wind_speed_da,
        'thermal_mean_altitude_achievement_percent':
        ds['altitude_achievement_percent'].mean(
            dim=['episode', 'start_distance']),
        'thermal_std_altitude_achievement_percent':
        ds['altitude_achievement_percent'].std(
            dim=['episode', 'start_distance'])
    })

    by_distance_ds = xr.Dataset({
        'thermal_mean_wind_speed':
        thermal_mean_wind_speed_da,
        'thermal_mean_altitude_achievement_percent':
        ds['altitude_achievement_percent'].mean(dim=['episode']),
        'thermal_std_altitude_achievement_percent':
        ds['altitude_achievement_percent'].std(dim=['episode'])
    })
    by_distance_ds = by_distance_ds.assign_coords(
        start_distance=by_distance_ds.coords['start_distance'].astype(float))

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):

        cm_to_inch = 1 / 2.54
        fig, ax = plt.figure(figsize=(6 * cm_to_inch, 5 * cm_to_inch),
                             layout='constrained'), plt.gca()

        cmap_name = 'viridis'
        colors: ListedColormap = plt.cm.get_cmap(cmap_name, 2)
        for (setup, color) in zip(['student_alone', 'student_with_birds'],
                                  colors.colors):
            by_distance_ds.sel(setup=setup).plot.scatter(
                x='thermal_mean_wind_speed',
                cmap=cmap_name,
                y='thermal_mean_altitude_achievement_percent',
                color=color,
                markersize='start_distance',
                ax=ax,
                alpha=0.1)
        distance_mean_ds.plot.scatter(
            x='thermal_mean_wind_speed',
            y='thermal_mean_altitude_achievement_percent',
            hue='setup',
            ax=ax,
            add_colorbar=False,
            add_legend=True,
            cmap=cmap_name)

        save_figure(
            plt.gcf(),
            output_dir=output_dir,
            name=
            'from_different_distances_using_train_glider__altitude_achievement_percent_for_episodes_for_different_wind_speeds',
            svg=True,
            png=True)


def _plot_wind_speed(ds: xr.Dataset, output_dir: Path):
    # calculate average horizontal wind for each thermal and episode
    _ADDITIONAL_RC_PARAMS = {}

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):

        thermal_mean_wind_speed_da = ds[
            'horizontal_air_velocity_earth_m_s'].mean(
                dim=['setup', 'start_distance', 'episode', 'time_s'])
        thermal_mean_wind_speed_da.attrs[
            'long_name'] = 'Mean horizontal wind speed'
        thermal_mean_wind_speed_da.attrs['standard_name'] = 'wind_speed'
        thermal_mean_wind_speed_da.attrs['units'] = 'm s-1'
        plt.bar(x=thermal_mean_wind_speed_da['thermal'],
                height=thermal_mean_wind_speed_da)
        plt.xlabel('thermal')
        plt.ylabel('Mean horizontal speed m s-1')

        save_figure(
            plt.gcf(),
            output_dir=output_dir,
            name=
            'average_wind_speed_per_thermal_along_setup_start_distance_and_episode',
        )


def _plot_wind_speed_with_achievement_percent(ds: xr.Dataset,
                                              output_dir: Path):

    _ADDITIONAL_RC_PARAMS = {
        'savefig.bbox': None,  # the default 'tight' caused issues
    }

    with select_plot_style('science',
                           _ADDITIONAL_RC_PARAMS,
                           engine_override='Agg'):

        subset_ds = ds[[
            'horizontal_air_velocity_earth_m_s', 'altitude_achievement_percent'
        ]]

        thermal_groupby = subset_ds.groupby('thermal', squeeze=False)
        nthermal = len(thermal_groupby.groups)

        ncols = 2
        nrows = int(np.ceil(nthermal / ncols))

        # create the grid
        fig = plt.figure(figsize=(4.75, 6.))
        gs = gridspec.GridSpec(nrows=nrows,
                               ncols=ncols,
                               figure=fig,
                               wspace=0.6,
                               hspace=0.6)
        gs.update(left=0.1, right=0.9, top=0.945, bottom=0.06)

        legend_handles = None
        legend_labels = None

        # sort the thermals by mean wind speed
        thermal_wind_means = subset_ds[
            'horizontal_air_velocity_earth_m_s'].mean(
                dim=['setup', 'start_distance', 'episode', 'time_s'])

        sorted_indices = thermal_wind_means.argsort().values
        ordered = thermal_wind_means.coords['thermal'].values[sorted_indices]

        subset_ds = subset_ds.sel(thermal=ordered)

        for i, label in enumerate(ordered):

            thermal_ds = subset_ds.sel(thermal=label)
            row, col = divmod(i, ncols)

            # create the divider
            pos = gs[row, col].get_position(fig)
            fig.text(pos.x0 + pos.width / 2,
                     pos.y1 + 0.02,
                     label,
                     ha='center',
                     va='bottom')

            horiz = [Size.Scaled(0.75), Size.Fixed(0.1), Size.Scaled(0.25)]
            vert = [Size.Scaled(1)]
            divider = Divider(fig, pos.bounds, horiz, vert, aspect=False)

            placeholder_ax = fig.add_axes(divider.get_position(),
                                          axes_locator=divider.new_locator(
                                              nx=0, ny=0))
            wind_ax = fig.add_axes(divider.get_position(),
                                   axes_locator=divider.new_locator(nx=2,
                                                                    ny=0))
            placeholder_ax.set_ylabel('Start distance $(\\mathrm{m})$',
                                      labelpad=20)
            placeholder_ax.set_xticks([])
            placeholder_ax.set_yticks([])
            placeholder_ax.set_xticklabels([])
            placeholder_ax.set_yticklabels([])

            # Hide spines
            for spine in placeholder_ax.spines.values():
                spine.set_visible(False)

            fig.canvas.draw()

            default_palette = sns.color_palette()

            df = thermal_ds.to_dataframe()

            wind_ax.yaxis.tick_right()
            wind_ax.yaxis.set_label_position('right')
            wind_ax.set_ylim(0., 10.)
            sns.boxenplot(data=df,
                          x='thermal',
                          y='horizontal_air_velocity_earth_m_s',
                          ax=wind_ax,
                          width=0.7,
                          flier_kws=dict(s=4, linewidths=0.5),
                          color=default_palette[4])

            wind_ax.set_xlabel('')
            wind_ax.set_xticks([])
            wind_ax.set_ylabel('Wind Speed $(\\mathrm{m ~ s}^{-1})$',
                               rotation=-90.,
                               labelpad=10)

            distances = thermal_ds.coords['start_distance']
            ndistances = len(distances)

            horiz = [Size.Scaled(1)]
            vert = [[Size.Scaled(1), Size.Fixed(0.05)]
                    for _ in range(ndistances)]
            vert = list(chain.from_iterable(vert))[:-1]  # flatten

            histogram_divider = Divider(fig,
                                        placeholder_ax.get_position().bounds,
                                        horiz,
                                        vert,
                                        aspect=False)

            histogram_axes: list[matplotlib.axes.Axes] = []

            for i, distance in enumerate(distances):
                distance_value = distance.values.item()
                pos = histogram_divider.get_position()
                assert isinstance(pos, tuple)

                histogram_ax = fig.add_axes(
                    pos,
                    axes_locator=histogram_divider.new_locator(nx=0, ny=i * 2))
                histogram_axes.append(histogram_ax)

                # configure the spines and ticks
                histogram_ax.set_yticks([])
                if i > 0:
                    sns.despine(ax=histogram_ax,
                                left=True,
                                bottom=False,
                                top=True,
                                right=True)
                    histogram_ax.set_xticks([])
                else:
                    sns.despine(ax=histogram_ax,
                                left=True,
                                bottom=False,
                                top=True,
                                right=True)
                    histogram_ax.xaxis.set_major_locator(
                        ticker.MultipleLocator(20))
                    histogram_ax.xaxis.set_minor_locator(
                        ticker.MultipleLocator(10))
                    histogram_ax.tick_params(top=False, which='both')
                    histogram_ax.set_xlabel('Altitude gain $(\\%)$')

            # plot using the data
            altitude_achievement_percent_da = thermal_ds[
                'altitude_achievement_percent']

            tick_font = None
            for i, distance in enumerate(distances):
                current_distance_da = altitude_achievement_percent_da.sel(
                    start_distance=distance)

                df = current_distance_da.to_dataframe()

                ticks = np.arange(0, 101, 10)

                histogram_ax = histogram_axes[i]
                distance_value = distance.values.item()
                histogram_ax.set_ylabel(f'${distance_value}$',
                                        rotation=0.,
                                        labelpad=10)

                palette = {
                    'student_alone': default_palette[3],
                    'student_with_birds': default_palette[0]
                }

                sns.histplot(data=df,
                             x='altitude_achievement_percent',
                             hue='setup',
                             ax=histogram_ax,
                             multiple='dodge',
                             legend=True,
                             bins=ticks,
                             edgecolor='lightgray',
                             linewidth=0,
                             palette=palette)

                histogram_ax.set_xlim(-5, 105)

                if i == 0:
                    # save the legend values
                    legend = histogram_ax.get_legend()
                    legend_handles = legend.legend_handles
                    legend_labels = [
                        text.get_text() for text in legend.get_texts()
                    ]
                    histogram_ax.get_legend().remove()

                    # save the tick font
                    tick_font = histogram_ax.get_xticklabels(
                    )[0].get_fontproperties()

                else:
                    histogram_ax.get_legend().remove()

                # use the same font for y labels as for x ticks
                assert tick_font is not None
                histogram_ax.yaxis.label.set_fontproperties(tick_font)

        assert legend_handles is not None
        assert legend_labels is not None
        legend_labels = [
            s.replace('student_alone', 'AI alone') for s in legend_labels
        ]
        legend_labels = [
            s.replace('student_with_birds', 'AI with Birds')
            for s in legend_labels
        ]
        fig.legend(handles=legend_handles,
                   labels=legend_labels,
                   loc='lower right')

        save_figure(
            fig,
            output_dir=output_dir,
            name=
            'from_different_distances_using_train_glider__wind_speed_and_achievement_percent',
        )


def _plot_thermal_vertical_velocity(ds: xr.Dataset, output_dir: Path):
    _ADDITIONAL_RC_PARAMS = {}

    vertical_velocity_ds = ds[['air_velocity_earth_m_per_s_z']]
    vertical_velocity_da = vertical_velocity_ds['air_velocity_earth_m_per_s_z']

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):
        df = vertical_velocity_ds.to_dataframe()

        sns.boxenplot(
            data=df,
            x='thermal',
            y='air_velocity_earth_m_per_s_z',
            # ax=wind_ax,
            width=0.7,
            flier_kws=dict(s=4, linewidths=0.5))

        save_figure(
            plt.gcf(),
            output_dir=output_dir,
            name='average_thermal_vertical_velocity_per_thermal',
        )

    # save stats
    reasonable_range = [-2., 5.]
    reasonable_da = (vertical_velocity_da >= reasonable_range[0]) & (
        vertical_velocity_da <= reasonable_range[1])

    dims = ['setup', 'start_distance', 'episode', 'time_s']

    count = vertical_velocity_da.count(dim=dims)
    reasonable_count = reasonable_da.sum(dim=dims)

    stat_ds = xr.Dataset(
        dict(min=vertical_velocity_da.min(dim=dims),
             max=vertical_velocity_da.max(dim=dims),
             mean=vertical_velocity_da.mean(dim=dims),
             median=vertical_velocity_da.median(dim=dims),
             count=count,
             reasonable_count=reasonable_count,
             non_reasonable_count=count - reasonable_count,
             non_reasonable_percent=(count - reasonable_count) / count * 100))

    write_dataframe(stat_ds.to_dataframe(),
                    output_dir=output_dir,
                    name='thermal_vertical_velocity')


def _plot_thermal_bird_vertical_velocity(output_dir: Path):
    _ADDITIONAL_RC_PARAMS = {}

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):

        bird_dataset_path = Path(
            'data/bird_comparison/processed/stork_trajectories_as_observation_log/merged_observation_log.csv'
        )

        ds = load_bird_dataset(bird_dataset_path)

        thermal_mean_vertical_velocity_da = ds[
            'velocity_earth_m_per_s_z'].mean(
                dim=['bird_name', 'episode', 'time_s']).assign_attrs(
                    units='m s-1')

        plt.bar(x=thermal_mean_vertical_velocity_da['thermal'],
                height=thermal_mean_vertical_velocity_da)

        plt.xlabel('thermal')
        plt.ylabel('Mean vertical speed m s-1')

        save_figure(
            plt.gcf(),
            output_dir=output_dir,
            name='bird_average_vertical_velocity_per_thermal',
        )


def main(policy_neptune_run_id: str, episode_count: int,
         r_distances: list[int]):
    configure_logger()
    ds = load_per_distance_dataset(policy_neptune_run_id=policy_neptune_run_id,
                                   episode_count=episode_count,
                                   r_distances=r_distances)

    generate_wind_speed_analysis(ds)


if __name__ == '__main__':
    tyro.cli(main)
