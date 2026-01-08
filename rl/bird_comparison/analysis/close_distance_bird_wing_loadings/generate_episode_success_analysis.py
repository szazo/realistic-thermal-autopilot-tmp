import logging
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from ..common.dataset_utils import save_figure, write_dataframe
from ..common.add_altitude_achievement_percent import add_agent_initial_and_maximum_altitude, add_altitude_achievement_percent
from .load_close_distance_dataset import load_close_distance_dataset
from utils.logging import configure_logger

logger = logging.getLogger('generate_episode_success_analysis')


def generate_episode_success_analysis(
    ds: xr.Dataset,
    output_dir: Path = Path(
        'results/analysis/from_close_distance_using_bird_wing_loadings/episode_success'
    )):
    logger.debug('preprocessing data...')
    ds = add_agent_initial_and_maximum_altitude(ds)
    ds = add_altitude_achievement_percent(
        ds,
        bird_maximum_altitude_reference_variable=
        'thermal_bird_maximum_altitude_per_bird')

    logger.debug('done')

    ds = _add_episode_success(ds)

    _generate_per_thermal_and_per_bird_success_analysis(ds,
                                                        output_dir=output_dir /
                                                        'per_thermal_per_bird')
    _generate_per_thermal_analysis(ds, output_dir=output_dir / 'per_thermal')
    _plot_altitude_achievement_percent_per_thermal_per_bird(
        ds, output_dir=output_dir / 'per_thermal_per_bird' / 'plots')


def _plot_altitude_achievement_percent_per_thermal_per_bird(
        ds: xr.Dataset, output_dir: Path):

    altitude_achievement_percent_episode_mean = ds[
        'altitude_achievement_percent'].mean(dim='episode')

    altitude_achievement_percent_episode_mean.plot.scatter(x='setup',
                                                           hue='bird_name',
                                                           col='thermal',
                                                           col_wrap=3)

    save_figure(
        plt.gcf(),
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__altitude_achievement_percent'
    )

    altitude_achievement_percent_episode_mean = altitude_achievement_percent_episode_mean.assign_coords(
        bird_name=altitude_achievement_percent_episode_mean.
        coords['bird_name'].astype(str))

    altitude_achievement_percent_episode_mean.plot.pcolormesh(x='thermal',
                                                              y='bird_name',
                                                              col='setup',
                                                              col_wrap=3)
    plt.gcf().set_figheight(10)

    save_figure(
        plt.gcf(),
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__altitude_achievement_percent_colormesh'
    )


def _generate_per_thermal_and_per_bird_success_analysis(
        ds: xr.Dataset, output_dir: Path):

    # student alone
    student_alone_episode_success_count_da = _create_episode_success_count_table(
        ds['episode_success'].sel(setup='student_alone'))
    df = student_alone_episode_success_count_da.to_pandas().transpose()
    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__student_alone_episode_success_count'
    )

    # student with birds
    student_with_birds_episode_success_count_da = _create_episode_success_count_table(
        ds['episode_success'].sel(setup='student_with_birds'))
    df = student_with_birds_episode_success_count_da.to_pandas().transpose()
    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__student_with_birds_episode_success_count'
    )

    # birds
    birds_episode_success_count_da = _create_episode_success_count_table(
        ds['episode_success'].sel(setup='birds'))
    df = birds_episode_success_count_da.to_pandas().transpose()
    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__birds_episode_success_count'
    )


def _generate_per_thermal_analysis(ds: xr.Dataset, output_dir: Path):
    # episode success percent
    df = ds['episode_success_percent'].to_pandas().transpose()
    assert isinstance(df, pd.DataFrame)
    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__episode_success_percent'
    )

    # episode success count
    df = ds['episode_success_count'].to_pandas().transpose()
    assert isinstance(df, pd.DataFrame)
    write_dataframe(
        df,
        output_dir=output_dir,
        name=
        'from_close_distance_using_bird_wing_loadings__episode_success_count')

    # episode count
    df = ds['episode_count'].to_pandas().transpose()
    assert isinstance(df, pd.DataFrame)
    write_dataframe(
        df,
        output_dir=output_dir,
        name='from_close_distance_using_bird_wing_loadings__episode_count')


def _create_episode_success_count_table(
        episode_success_source_da: xr.DataArray):

    episode_success_all_nan_da = episode_success_source_da.isnull().all(
        dim=['episode'])
    episode_success_count_da = episode_success_source_da.sum(dim=['episode'],
                                                             skipna=True)

    episode_success_count_da = xr.where(~episode_success_all_nan_da,
                                        episode_success_count_da, np.nan)

    return episode_success_count_da


def _add_episode_success(ds: xr.Dataset):
    episode_success_da = ds['altitude_achievement_percent'] == 100.
    episode_success_nan_da = ds['altitude_achievement_percent'].isnull(
    )  #.all(dim='episode')
    episode_success_da = xr.where(~episode_success_nan_da, episode_success_da,
                                  np.nan)

    ds = ds.assign(episode_success=episode_success_da)

    # count and percent
    episode_success_count_da = ds['episode_success'].sum(
        dim=['bird_name', 'episode'], skipna=True)
    episode_count_da = ds['episode_success'].notnull().sum(
        dim=['bird_name', 'episode'])

    episode_success_percent_da = (episode_success_count_da /
                                  episode_count_da) * 100

    ds = ds.assign(episode_success_count=episode_success_count_da)
    ds = ds.assign(episode_count=episode_count_da)
    ds = ds.assign(episode_success_percent=episode_success_percent_da)
    return ds


def main(policy_neptune_run_id: str, episode_count: int):
    configure_logger()
    ds = load_close_distance_dataset(
        policy_neptune_run_id=policy_neptune_run_id,
        episode_count=episode_count)

    generate_episode_success_analysis(ds)


if __name__ == '__main__':
    tyro.cli(main)
