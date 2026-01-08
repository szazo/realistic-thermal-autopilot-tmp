import logging
from typing import cast, Any, Literal
from pathlib import Path
import scipy.stats as stats
from utils.plot_style import select_plot_style
import xarray as xr
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import tyro
import seaborn as sns
import pandas as pd
import sigfig
from mpl_toolkits.axes_grid1 import Divider, Size

from bird_comparison.analysis.common.dataset_utils import (
    save_figure, format_p_star, write_dataframe,
    calculate_standard_error_of_mean,
    calculate_95_confidence_interval_from_sem)
from bird_comparison.analysis.common.add_altitude_achievement_percent import (
    add_agent_initial_and_maximum_altitude, add_altitude_achievement_percent)
from bird_comparison.analysis.common.thermal_map import thermal_colormap
from bird_comparison.analysis.close_distance_bird_wing_loadings.load_close_distance_dataset import load_close_distance_dataset
from utils.logging import configure_logger


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
@dataclass
class LinearRegressionResult:
    slope: float
    intercept: float
    pearson_r: float
    p: float
    slope_stderr: float
    intercept_stderr: float


logger = logging.getLogger('generate_vertical_velocity_analysis')


def generate_vertical_velocity_analysis(
    ds: xr.Dataset,
    n_boot: int,
    output_dir: Path = Path(
        'results/analysis/from_close_distance_using_bird_wing_loadings/vertical_velocity'
    )):

    logger.debug('preprocessing data...')
    ds = add_agent_initial_and_maximum_altitude(ds)
    ds = add_altitude_achievement_percent(
        ds,
        bird_maximum_altitude_reference_variable=
        'thermal_bird_maximum_altitude_per_bird')

    logger.debug('done')

    _plot_thermal_vertical_velocity(ds, output_dir)

    for only_success in [True, False]:

        success_suffix = '_only_success_episodes' if only_success else ''

        # generate scatter plot and linear regression analysis
        for setup in ['student_alone', 'student_with_birds']:

            with_stat_ds = _create_vertical_velocity_comparison_ds(
                current_ds=ds, only_success=only_success, n_boot=n_boot)

            # scatter plot
            scatter_name = f'from_close_distance_using_bird_wing_loadings__{setup}_vs_birds_vertical_velocity_mean{success_suffix}'

            bird_ds = with_stat_ds.sel(setup='birds', drop=True)
            ai_ds = with_stat_ds.sel(setup=setup, drop=True)

            _plot_vertical_velocity_scatter(bird_ds=bird_ds,
                                            ai_ds=ai_ds,
                                            output_dir=output_dir / 'scatter',
                                            name=scatter_name)

            # linregress
            linregress_name = f'from_close_distance_using_bird_wing_loadings__{setup}_vs_birds_vertical_velocity_linregress{success_suffix}'
            _generate_linregress_analysis(bird_ds=bird_ds,
                                          ai_ds=ai_ds,
                                          output_dir=output_dir / 'linregress',
                                          name=linregress_name)

        # violinplot
        violin_name = f'from_close_distance_using_bird_wing_loadings__student_vs_birds_vertical_velocity_distribution{success_suffix}'

        violin_ds = _filter_success_episodes(ds) if only_success else ds
        _plot_vertical_velocity_violin(ds=violin_ds,
                                       output_dir=output_dir / 'violin',
                                       name=violin_name)


def _generate_linregress_analysis(bird_ds: xr.Dataset, ai_ds: xr.Dataset,
                                  output_dir: Path, name: str):

    student_n = ai_ds['vertical_velocity_mean'].count().item()
    bird_n = bird_ds['vertical_velocity_mean'].count().item()

    a = bird_ds['vertical_velocity_mean']
    b = ai_ds['vertical_velocity_mean']

    linregress_result = _calculate_linregress(a.values, b.values)
    linregress_df = pd.DataFrame([asdict(linregress_result)])

    linregress_df['p_value_fmt'] = linregress_df['p'].apply(format_p_star)
    linregress_df['p'] = linregress_df['p'].apply(sigfig.round,
                                                  sigfigs=3,
                                                  notation='sci')
    linregress_df['bird_n'] = bird_n
    linregress_df['student_n'] = student_n

    write_dataframe(linregress_df, output_dir=output_dir, name=name)


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

    dims = ['setup', 'bird_name', 'episode', 'time_s']

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


def _plot_vertical_velocity_violin(ds: xr.Dataset, output_dir: Path,
                                   name: str):

    vertical_velocity_da = ds['velocity_earth_m_per_s_z']
    df = vertical_velocity_da.to_dataframe()

    _ADDITIONAL_RC_PARAMS = {
        'xtick.top': False,
        'axes.spines.top': False,
        'ytick.right': False,
        'axes.spines.right': False,
    }

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()

        sns.violinplot(data=df,
                       x='thermal',
                       y='velocity_earth_m_per_s_z',
                       hue='setup',
                       density_norm='width',
                       cut=0,
                       inner='box',
                       ax=ax)

        sns.stripplot(data=df,
                      x='thermal',
                      y='velocity_earth_m_per_s_z',
                      hue='setup',
                      dodge=True,
                      size=0.5,
                      jitter=True,
                      zorder=1,
                      ax=ax,
                      legend=False)

        save_figure(fig, output_dir=output_dir, name=name)


def _plot_vertical_velocity_scatter(bird_ds: xr.Dataset, ai_ds: xr.Dataset,
                                    output_dir: Path, name: str):
    _ADDITIONAL_RC_PARAMS = {
        'xtick.top': False,
        'axes.spines.top': False,
        'ytick.right': False,
        'axes.spines.right': False,
    }

    wl_min = ai_ds['wing_loading'].min().item()
    wl_max = ai_ds['wing_loading'].max().item()

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):

        cm_to_inch = 1 / 2.54

        axis_width = 3.5 * cm_to_inch
        axis_height = 3.5 * cm_to_inch

        horizontal = [
            Size.Scaled(0.5),
            Size.Fixed(axis_width),
            Size.Scaled(0.5)
        ]
        vertical = [
            Size.Scaled(0.5),
            Size.Fixed(axis_height),
            Size.Scaled(0.5)
        ]

        fig = plt.figure(figsize=(1, 1))
        divider = Divider(fig,
                          pos=(0, 0, 1, 1),
                          horizontal=horizontal,
                          vertical=vertical,
                          aspect=False)

        ax = fig.add_axes(divider.get_position(),
                          axes_locator=divider.new_locator(nx=1, ny=1))

        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim((0, 2.))
        ax.set_ylim((0, 2.))

        ax.set_xlabel(
            'Bird vertical speed, $V_{Z}^{Bird}~(\\mathrm{m ~ s}^{-1})$')
        ax.set_ylabel('AI vertical speed, $V_{Z}^{AI}~(\\mathrm{m ~ s}^{-1})$')

        ax.plot(ax.get_xlim(),
                ax.get_ylim(),
                color='grey',
                linestyle='--',
                linewidth=1.0,
                label='x=y',
                zorder=0)

        thermal_cmap = thermal_colormap()

        use_as_error_bar: Literal['iqr', 'bootstrapped_percentile'] = 'iqr'

        for thermal in bird_ds.coords['thermal'].values:
            cc = thermal_cmap[thermal]

            rgba_color = mcolors.to_rgba(cc, alpha=0.7)

            current_bird_ds = bird_ds.sel(thermal=thermal)
            current_ai_ds = ai_ds.sel(thermal=thermal)

            x_da = current_bird_ds['vertical_velocity_mean']
            y_da = current_ai_ds['vertical_velocity_mean']

            alpha = (current_ai_ds['wing_loading'] - wl_min) / (wl_max -
                                                                wl_min)

            # currently not used, alpha based on wing loadings
            rgba_colors = [
                mcolors.to_rgba(cc, alpha=0.2 + 0.8 * c_alpha)
                for c_alpha in alpha.values
            ]

            if use_as_error_bar == 'bootstrapped_percentile':
                # 95% bootstrap percentile CI for the mean.
                x_low = current_bird_ds['percentile_95_low']
                x_high = current_bird_ds['percentile_95_high']

                y_low = current_ai_ds['percentile_95_low']
                y_high = current_ai_ds['percentile_95_high']
            elif use_as_error_bar == 'iqr':
                x_low = current_bird_ds['q1']
                x_high = current_bird_ds['q3']
                y_low = current_ai_ds['q1']
                y_high = current_ai_ds['q3']

            x_low_error = x_da - x_low
            x_high_error = x_high - x_da

            y_low_error = y_da - y_low
            y_high_error = y_high - y_da

            x_error = np.stack((x_low_error, x_high_error), axis=0)
            y_error = np.stack((y_low_error, y_high_error), axis=0)

            markersize = 14
            ax.scatter(
                x=x_da,
                y=y_da,
                label=thermal,
                color=rgba_color,
                edgecolors='w',
                linewidths=0.5,  # line width for the edges
                s=markersize,
                zorder=2)

            # continue
            ax.errorbar(
                x=x_da,
                y=y_da,
                xerr=x_error,
                # yerr=y_error_da,
                fmt='none',
                ecolor=cc,
                alpha=0.1,
                capsize=1.,
                zorder=1,
                elinewidth=0.5,
                capthick=0.5)

            ax.errorbar(
                x=x_da,
                y=y_da,
                # xerr=x_error_da,
                yerr=y_error,
                fmt='none',
                ecolor=cc,
                alpha=0.1,
                zorder=1,
                capsize=1.,
                elinewidth=0.5,
                capthick=0.5)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))

        save_figure(fig, output_dir=output_dir, name=name)

        plt.show()


def _filter_success_episodes(ds: xr.Dataset):
    ds = ds.where(ds['altitude_achievement_percent'] == 100, drop=True)

    return ds


def _calculate_bootstrap_mean_percentiles(x, n_boot, q=[2.5, 97.5]):

    # merge the last to axes
    x = x.reshape(*x.shape[:-2], -1)

    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.array([np.nan, np.nan])

    # calculate the mean of the sample n_boot times
    # resample items with replacement
    sample_list = [
        np.mean(np.random.choice(x, size=len(x), replace=True))
        for _ in range(n_boot)
    ]
    samples = np.array(sample_list)

    # calculate the percentiles
    percentiles = np.percentile(samples, q)

    return percentiles


def _percentile_interval_with_bootstrapping(da: xr.DataArray, n_boot: int):

    percentile_da = xr.apply_ufunc(
        _calculate_bootstrap_mean_percentiles,
        da,
        input_core_dims=[['episode', 'time_s']],
        output_core_dims=[['percentile_95_interval']],
        vectorize=True,
        kwargs={'n_boot': n_boot})
    percentile_da = percentile_da.assign_coords(
        percentile_95_interval=['low', 'high'])

    percentile_low = percentile_da.sel(percentile_95_interval='low', drop=True)
    percentile_high = percentile_da.sel(percentile_95_interval='high',
                                        drop=True)

    return percentile_low, percentile_high


def _create_vertical_velocity_comparison_ds(current_ds: xr.Dataset,
                                            only_success: bool, n_boot: int):

    if only_success:
        # filter only success episodes
        current_ds = _filter_success_episodes(current_ds)

    def assign_wing_loading(bird_ds: xr.DataArray):
        wing_loading = bird_ds['mass_kg'] / bird_ds['wing_area_m2']
        flat = wing_loading.values.ravel()
        first = flat[~np.isnan(flat)][0]

        bird_ds = bird_ds.assign(wing_loading=first)
        return bird_ds

    current_ds = current_ds.groupby(
        'bird_name', squeeze=False).apply(lambda x: assign_wing_loading(x))

    # calculate mean vertical velocity by thermal and bird_name
    vertical_velocity_da = current_ds['velocity_earth_m_per_s_z']

    # mean and standard error
    vertical_velocity_mean_da = vertical_velocity_da.mean(
        dim=['episode', 'time_s']).assign_attrs(units='m s-1')
    vertical_velocity_std_da = vertical_velocity_da.std(
        dim=['episode', 'time_s']).assign_attrs(units='m s-1')
    current_ds = current_ds.assign(
        vertical_velocity_mean=vertical_velocity_mean_da)
    current_ds = current_ds.assign(
        vertical_velocity_std=vertical_velocity_std_da)

    # standard error of mean and 95% confidence interval of the mean
    n = current_ds['velocity_earth_m_per_s_z'].count(dim=['episode', 'time_s'])
    sem_da = calculate_standard_error_of_mean(std_da=vertical_velocity_std_da,
                                              n_for_std_da=n)
    ci95_da = calculate_95_confidence_interval_from_sem(sem_da)
    current_ds = current_ds.assign(vertical_velocity_ci95=ci95_da)

    # bootstrapped 95% percentile of the mean
    percentile_low, percentile_high = _percentile_interval_with_bootstrapping(
        da=vertical_velocity_da, n_boot=n_boot)

    current_ds = current_ds.assign(percentile_95_low=percentile_low)
    current_ds = current_ds.assign(percentile_95_high=percentile_high)

    # q1 and q3 quantiles
    q1_da = vertical_velocity_da.quantile(0.25, dim=['episode', 'time_s'])
    q3_da = vertical_velocity_da.quantile(0.75, dim=['episode', 'time_s'])
    current_ds = current_ds.assign(q1=q1_da)
    current_ds = current_ds.assign(q3=q3_da)

    return current_ds


def _calculate_linregress(x: np.ndarray,
                          y: np.ndarray) -> LinearRegressionResult:

    # check nans are at the same indices
    nan_match = np.all(np.isnan(x) == np.isnan(y))
    assert nan_match == True

    mask = ~np.isnan(x)
    x_clean = x[mask]
    y_clean = y[mask]

    result = stats.linregress(x=x_clean, y=y_clean)
    result = cast(Any, result)

    return LinearRegressionResult(slope=result.slope,
                                  intercept=result.intercept,
                                  pearson_r=result.rvalue,
                                  p=result.pvalue,
                                  slope_stderr=result.stderr,
                                  intercept_stderr=result.intercept_stderr)


def main(policy_neptune_run_id: str, episode_count: int, n_bootstrap: int):
    configure_logger()
    ds = load_close_distance_dataset(
        policy_neptune_run_id=policy_neptune_run_id,
        episode_count=episode_count)

    generate_vertical_velocity_analysis(ds, n_boot=n_bootstrap)


if __name__ == '__main__':
    tyro.cli(main)
