from pathlib import Path
import logging
import matplotlib
from utils.logging import configure_logger
from utils.plot_style import select_plot_style
import xarray as xr
import numpy as np
import pandas as pd
import sigfig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes
from scipy import stats
from mpl_toolkits.axes_grid1 import Divider, Size

from bird_comparison.analysis.common.dataset_utils import (
    save_figure, write_dataframe, format_p_star,
    calculate_standard_error_of_mean,
    calculate_95_confidence_interval_from_sem)
from bird_comparison.analysis.common.thermal_map import thermal_colormap
from bird_comparison.analysis.common.add_altitude_achievement_percent import add_agent_initial_and_maximum_altitude, add_altitude_achievement_percent
from bird_comparison.analysis.different_distances_train_wing_loading.load_per_distance_dataset import load_per_distance_dataset
import tyro

logger = logging.getLogger('altitude_achievement_percent')
logger.setLevel(logging.DEBUG)


def _add_altitude_achievement_percent_episode_mean_std(ds: xr.Dataset):

    # by thermal and distance
    ds['altitude_achievement_percent_mean'] = ds[
        'altitude_achievement_percent'].mean(dim=['episode'])
    ds['altitude_achievement_percent_std'] = ds[
        'altitude_achievement_percent'].std(dim=['episode'])

    # by distance
    ds['altitude_achievement_percent_by_distance_mean'] = ds[
        'altitude_achievement_percent'].mean(dim=['thermal', 'episode'])
    ds['altitude_achievement_percent_by_distance_std'] = ds[
        'altitude_achievement_percent'].std(dim=['thermal', 'episode'])

    return ds


def _add_altitude_achievement_percent_episode_sem_95ci(ds: xr.Dataset):

    # by thermal and distance
    n = ds['altitude_achievement_percent'].count(dim='episode')

    # https://resources.nu.edu/statsresources/ComputingSEM
    sem = calculate_standard_error_of_mean(
        std_da=ds['altitude_achievement_percent_std'], n_for_std_da=n)
    ci95 = calculate_95_confidence_interval_from_sem(sem)

    ds['altitude_achievement_percent_sem'] = sem
    ds['altitude_achievement_percent_ci95'] = ci95

    # by distance
    n = ds['altitude_achievement_percent'].count(dim=['thermal', 'episode'])
    sem = calculate_standard_error_of_mean(
        std_da=ds['altitude_achievement_percent_by_distance_std'],
        n_for_std_da=n)
    ci95 = calculate_95_confidence_interval_from_sem(sem)
    ds['altitude_achievement_percent_by_distance_sem'] = sem
    ds['altitude_achievement_percent_by_distance_ci95'] = ci95

    return ds


def generate_altitude_achievement_percent_by_distance_analysis(
    ds: xr.Dataset,
    output_dir: Path = Path(
        'results/analysis/from_different_distances_using_train_glider/altitude_achievement_percent'
    )):

    logger.debug('preprocessing data...')
    ds = add_agent_initial_and_maximum_altitude(ds)
    ds = add_altitude_achievement_percent(
        ds,
        bird_maximum_altitude_reference_variable='thermal_bird_maximum_altitude'
    )
    ds = _add_altitude_achievement_percent_episode_mean_std(ds)
    ds = _add_altitude_achievement_percent_episode_sem_95ci(ds)

    logger.debug('done')

    _plot_altitude_achievement_percent_by_distance(ds, output_dir)
    _generate_t_test_alone_vs_with_birds_by_distance(ds, output_dir)
    _generate_t_test_alone_vs_with_birds_by_distance_and_thermal(
        ds, output_dir)
    _generate_achievement_percent_mean_std_by_distance_and_thermal(
        ds, output_dir)
    _generate_achievement_percent_mean_std_by_distance(ds, output_dir)


def _generate_achievement_percent_mean_std_by_distance_and_thermal(
        ds: xr.Dataset, output_dir: Path):

    df = ds[[
        'altitude_achievement_percent_mean',
        'altitude_achievement_percent_std', 'altitude_achievement_percent_ci95'
    ]].to_dataframe(dim_order=['start_distance', 'thermal', 'setup'])

    sf = 3
    df['altitude_achievement_percent_mean_pm_std'] = df.apply(
        lambda row:
        f'{sigfig.round(row["altitude_achievement_percent_mean"], sigfigs=sf)} ± {sigfig.round(row["altitude_achievement_percent_std"], sigfigs=sf)}',
        axis=1)

    df['altitude_achievement_percent_mean_pm_ci95'] = df.apply(
        lambda row:
        f'{sigfig.round(row["altitude_achievement_percent_mean"], sigfigs=sf)} ± {sigfig.round(row["altitude_achievement_percent_ci95"], sigfigs=sf)}',
        axis=1)

    df = df.reset_index().pivot_table(
        index=['start_distance', 'thermal'],
        columns='setup',
        values=[
            'altitude_achievement_percent_mean_pm_ci95',
            'altitude_achievement_percent_mean_pm_std'
        ],
        aggfunc='first')

    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__altitude_achievement_percent_mean_std'
    )


def _generate_achievement_percent_mean_std_by_distance(ds: xr.Dataset,
                                                       output_dir: Path):

    df = ds[[
        'altitude_achievement_percent_by_distance_mean',
        'altitude_achievement_percent_by_distance_std',
        'altitude_achievement_percent_by_distance_ci95'
    ]].to_dataframe(dim_order=['start_distance', 'setup'])

    sf = 3

    df['altitude_achievement_percent_mean_pm_std'] = df.apply(
        lambda row:
        f'{sigfig.round(row["altitude_achievement_percent_by_distance_mean"], sigfigs=sf)} ± {sigfig.round(row["altitude_achievement_percent_by_distance_std"], sigfigs=sf)}',
        axis=1)

    df['altitude_achievement_percent_mean_pm_ci95'] = df.apply(
        lambda row:
        f'{sigfig.round(row["altitude_achievement_percent_by_distance_mean"], sigfigs=sf)} ± {sigfig.round(row["altitude_achievement_percent_by_distance_ci95"], sigfigs=sf)}',
        axis=1)

    df = df.reset_index().pivot_table(
        index=['start_distance'],
        columns='setup',
        values=[
            'altitude_achievement_percent_mean_pm_ci95',
            'altitude_achievement_percent_mean_pm_std'
        ],
        aggfunc='first')

    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__altitude_achievement_percent_by_distance_mean_std'
    )


def _generate_t_test_alone_vs_with_birds_by_distance_and_thermal(
        ds: xr.Dataset, output_dir: Path):

    student_alone_episode_achievement_percent_da = _student_alone_episode_achievement_percent(
        ds)
    student_with_birds_episode_achievement_percent_da = _student_with_birds_episode_achievement_percent(
        ds)

    ttest_da = xr.apply_ufunc(
        _independent_welch_ttest_func,
        student_alone_episode_achievement_percent_da,
        student_with_birds_episode_achievement_percent_da,
        input_core_dims=[['episode'], ['episode']],
        output_core_dims=[['stat_p']])
    assert isinstance(ttest_da, xr.DataArray)
    ttest_da = _independent_welch_ttest_assign_coords(ttest_da,
                                                      dim_name='stat_p')
    df = ttest_da.to_dataframe(
        name='t_stat', dim_order=['thermal', 'start_distance', 'stat_p'])
    df = df.drop(columns=['thermal_id'])
    df = _format_ttest_result_dataframe(df)

    def rename_columns(df: pd.DataFrame):
        df = _rename_ttest_result_columns(df)
        df = df.rename_axis(index=['Thermal', 'Start Distance (m)'],
                            columns=None)
        df = df.rename(
            columns={
                'mean1_std1_fmt': 'student alone mean\\textpm std',
                'mean2_std2_fmt': 'student with birds mean\\textpm std'
            })

        return df

    write_dataframe(
        rename_columns(df),
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__student_alone_vs_student_with_birds_t_test_per_thermal_all_columns'
    )

    # select specific columns
    df = df[[
        'mean1_std1_fmt', 'mean2_std2_fmt', 't_stat', 'p_value_fmt', 'cohen_d',
        'df'
    ]]
    assert isinstance(df, pd.DataFrame)

    write_dataframe(
        rename_columns(df),
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__student_alone_vs_student_with_birds_t_test_per_thermal'
    )


def _rename_ttest_result_columns(df: pd.DataFrame):

    return df.rename(
        columns={
            't_stat': 't',
            'p_value': 'P (unformatted)',
            'p_value_fmt': 'P',
            'df': 'DOF',
            'levene_p': 'Levene\'s test (P)',
            'cohen_d': 'Cohen\'s d',
            'mean1_std1_fmt': 'mean\\textpm std',
            'mean2_std2_fmt': 'mean\\textpm std',
        })


def _generate_t_test_alone_vs_with_birds_by_distance(ds: xr.Dataset,
                                                     output_dir: Path):

    student_alone_episode_achievement_percent_da = _student_alone_episode_achievement_percent(
        ds)
    student_with_birds_episode_achievement_percent_da = _student_with_birds_episode_achievement_percent(
        ds)

    ttest_da = xr.apply_ufunc(
        _independent_ttest_func_for_last_two_dims,
        student_alone_episode_achievement_percent_da,
        student_with_birds_episode_achievement_percent_da,
        input_core_dims=[['thermal', 'episode'], ['thermal', 'episode']],
        output_core_dims=[['stat_p']])
    ttest_da = _independent_welch_ttest_assign_coords(ttest_da,
                                                      dim_name='stat_p')
    df = ttest_da.to_dataframe(name='paired_t_stat',
                               dim_order=['start_distance', 'stat_p'])

    df = _format_ttest_result_dataframe(df)

    def rename_columns(df: pd.DataFrame):
        df = _rename_ttest_result_columns(df)
        df = df.rename_axis(index=['Start Distance (m)'], columns=None)
        df = df.rename(
            columns={
                'mean1_std1_fmt': 'student alone mean\\textpm std',
                'mean2_std2_fmt': 'student with birds mean\\textpm std'
            })

        return df

    write_dataframe(
        rename_columns(df),
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__student_alone_vs_student_with_birds_t_test_all_columns'
    )

    # select specific columns
    df = df[[
        'mean1_std1_fmt', 'mean2_std2_fmt', 't_stat', 'p_value_fmt', 'cohen_d',
        'df'
    ]]
    assert isinstance(df, pd.DataFrame)

    write_dataframe(
        rename_columns(df),
        output_dir=output_dir,
        name=
        'from_different_distances_using_train_glider__student_alone_vs_student_with_birds_t_test'
    )


def _student_alone_episode_achievement_percent(ds: xr.Dataset):
    return ds['altitude_achievement_percent'].sel(setup='student_alone')


def _student_with_birds_episode_achievement_percent(ds: xr.Dataset):
    return ds['altitude_achievement_percent'].sel(setup='student_with_birds')


def _independent_ttest_func_for_last_two_dims(x, y):
    # merge the two last dims
    x = x.reshape(x.shape[:-2] + (-1, ))
    y = y.reshape(y.shape[:-2] + (-1, ))

    return _independent_welch_ttest_func(x, y)


def _independent_welch_ttest_assign_coords(ttest_da: xr.DataArray,
                                           dim_name: str):
    ttest_da = ttest_da.assign_coords({
        dim_name: [
            't_stat', 'p_value', 'df', 'n1', 'n2', 'mean1', 'mean2', 'std1',
            'std2', 'cohen_d', 'levene_p'
        ]
    })
    return ttest_da


def _independent_welch_ttest_func(x, y):

    stat, levene_p = stats.levene(x, y, axis=-1)

    test_result = stats.ttest_ind(x, y, axis=-1, equal_var=False)
    t = test_result.statistic
    p = test_result.pvalue
    df = test_result.df

    assert isinstance(t, np.ndarray)
    assert isinstance(p, np.ndarray)
    assert isinstance(df, np.ndarray)

    n1 = np.sum(~np.isnan(x), axis=-1)
    n2 = np.sum(~np.isnan(y), axis=-1)

    # calculate Cohen's d
    var1 = np.var(x, axis=-1, ddof=1)
    var2 = np.var(y, axis=-1, ddof=1)

    mean1 = np.mean(x, axis=-1)
    mean2 = np.mean(y, axis=-1)

    cohen_d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)

    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)

    res = np.stack(
        [t, p, df, n1, n2, mean1, mean2, std1, std2, cohen_d, levene_p],
        axis=-1)

    return res


def _format_ttest_result_dataframe(df: pd.DataFrame):
    df_unstack = df.unstack('stat_p')
    assert isinstance(df_unstack, pd.DataFrame)
    df = df_unstack

    df.columns = df.columns.get_level_values(
        1)  # remove the first level of columns
    assert isinstance(df, pd.DataFrame)
    df['p_value_fmt'] = df['p_value'].apply(format_p_star)
    df['t_stat'] = df['t_stat'].apply(sigfig.round, decimals=1)
    df['p_value'] = df['p_value'].apply(sigfig.round,
                                        sigfigs=3,
                                        notation='sci')
    df['df'] = df['df'].apply(sigfig.round, decimals=1)
    df['cohen_d'] = df['cohen_d'].apply(sigfig.round, decimals=2)
    df['levene_p'] = df['levene_p'].apply(sigfig.round,
                                          sigfigs=3,
                                          notation='sci')
    # cutoff=35 means that if the first two significant figures of std is larger than 35,
    # then it will use 1 significant figures for the uncertainty (and the mean)
    df['mean1_std1_fmt'] = df.apply(lambda row: sigfig.round(
        row['mean1'], uncertainty=row['std1'], cutoff=35),
                                    axis=1)
    df['mean2_std2_fmt'] = df.apply(lambda row: sigfig.round(
        row['mean2'], uncertainty=row['std2'], cutoff=35),
                                    axis=1)

    return df


def _plot_altitude_achievement_percent_by_distance(ds: xr.Dataset,
                                                   output_dir: Path):

    _ADDITIONAL_RC_PARAMS = {
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.top': False,
        # 'ytick.right': False,
        'axes.spines.top': False,
        'ytick.right': False,
        'axes.spines.right': False,
        # legend
        'legend.labelspacing': 0.3,
        'legend.handletextpad': 0.3,
        'legend.columnspacing': 1.0
    }

    with select_plot_style('science', _ADDITIONAL_RC_PARAMS):

        cm_to_inch = 1 / 2.54

        axis_width = 6 * cm_to_inch
        axis_height = 3.5 * cm_to_inch
        pad_width = 0.5 * cm_to_inch

        fig = plt.figure(figsize=(1, 1))

        horizontal = [
            Size.Scaled(0.5),
            Size.Fixed(axis_width),
            Size.Fixed(pad_width),
            Size.Fixed(axis_width),
            Size.Scaled(0.5)
        ]
        vertical = [
            Size.Scaled(0.5),
            Size.Fixed(axis_height),
            Size.Scaled(0.5)
        ]

        divider = Divider(fig,
                          pos=(0, 0, 1, 1),
                          horizontal=horizontal,
                          vertical=vertical,
                          aspect=False)

        ax1 = fig.add_axes(divider.get_position(),
                           axes_locator=divider.new_locator(nx=1, ny=1))
        ax2 = fig.add_axes(divider.get_position(),
                           axes_locator=divider.new_locator(nx=3, ny=1),
                           sharey=ax1)

        fig_axes = [ax1, ax2]

        start_distances = ds['altitude_achievement_percent_mean'].coords[
            'start_distance']

        namedict = [{
            'setup': 'student_alone'
        }, {
            'setup': 'student_with_birds'
        }]

        thermal_cmap = thermal_colormap()

        for ax, namedict in zip(fig_axes, namedict):

            setup = namedict['setup']

            assert isinstance(ax, matplotlib.axes.Axes)

            mean_data = ds['altitude_achievement_percent_mean'].loc[namedict]
            std_data = ds['altitude_achievement_percent_std'].loc[namedict]
            ci95_data = ds['altitude_achievement_percent_ci95'].loc[namedict]

            for thermal in mean_data.coords['thermal'].values:
                color = thermal_cmap[thermal]

                thermal_mean_da = mean_data.sel(thermal=thermal)
                thermal_ci95_da = ci95_data.sel(thermal=thermal)

                ax.plot(start_distances,
                        thermal_mean_da,
                        marker='o',
                        markersize=2.,
                        linewidth=1.,
                        label=thermal,
                        color=color)

                ax.fill_between(x=start_distances,
                                y1=thermal_mean_da - thermal_ci95_da,
                                y2=thermal_mean_da + thermal_ci95_da,
                                alpha=0.1,
                                color=color,
                                linewidth=0.)

            ax.yaxis.set_major_locator(ticker.MultipleLocator(20.))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

            ticks = ax.set_xticks(start_distances)

            # hide label for 75
            index = list(start_distances.values).index(75)
            ticks[index].label1.set_visible(False)

            ax.set_xlabel('Start distance $(\\mathrm{m})$')

            if setup == 'student_with_birds':
                ax.tick_params(labelleft=False)

            if setup == 'student_alone':
                ax.set_ylabel('Altitude gain $(\\%)$')

                ax.legend(loc='upper right', ncol=2)

            offset = 0.
            for thermal, thermal_std in std_data.groupby('thermal',
                                                         squeeze=False):
                thermal_std = thermal_std.squeeze(dim='thermal')

                x = thermal_std['start_distance'] + offset
                y = mean_data.sel(thermal=thermal)
                yerr = thermal_std.values

                if False:
                    ax.errorbar(x=x,
                                y=y,
                                yerr=yerr,
                                fmt='none',
                                ecolor='gray',
                                capsize=3,
                                alpha=0.6)

        save_figure(
            fig,
            output_dir=output_dir,
            name=
            'from_different_distances_using_train_glider__altitude_achievement_percent'
        )

        plt.show()


def main(policy_neptune_run_id: str, episode_count: int,
         r_distances: list[int]):
    configure_logger()

    ds = load_per_distance_dataset(policy_neptune_run_id=policy_neptune_run_id,
                                   episode_count=episode_count,
                                   r_distances=r_distances)

    generate_altitude_achievement_percent_by_distance_analysis(ds)


if __name__ == '__main__':
    tyro.cli(main)
