from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from ..common.thermal_map import remap_thermal_names
from ..common.dataset_utils import (filter_agent_id_pd, remove_thermal,
                                    resolve_thermal_info_from_bird_dataset2,
                                    filter_agent_id)


@dataclass
class Info:
    id: str
    start_distance_r_m: float


@dataclass(kw_only=True)
class DatasetPath(Info):
    filepath: Path | str


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    info: Info


def load_per_distance_dataset(policy_neptune_run_id: str, episode_count: int,
                              r_distances: list[int]):

    bird_dataset_path = Path(
        'data/bird_comparison/processed/stork_trajectories_as_observation_log/merged_observation_log.csv'
    )

    root_path = Path(
        'results/eval/realistic/peer_informed/from_different_distances_using_train_glider'
    )

    cache_dir = Path('data/cache/from_different_distances_using_train_glider')
    cache_filepath = cache_dir / f'from_different_distances_using_train_glider{"_".join(map(str, r_distances))}_{episode_count}.nc'

    if not cache_filepath.exists():

        print('cache file does not exists, loading .csvs...')

        agent_id = 'student0'

        # student_alone
        student_alone_paths = _resolve_dataset_paths(
            r_distances=r_distances,
            episode_count=episode_count,
            policy_neptune_run_id=policy_neptune_run_id,
            experiment_setup='student_alone')
        student_alone_ds = _load_and_merge_datasets(
            datasets=student_alone_paths,
            agent_id=agent_id,
            base_path=root_path)
        assert student_alone_ds is not None

        # student_with_birds
        student_with_birds_paths = _resolve_dataset_paths(
            r_distances=r_distances,
            episode_count=episode_count,
            policy_neptune_run_id=policy_neptune_run_id,
            experiment_setup='student_with_birds')
        student_with_birds_ds = _load_and_merge_datasets(
            student_with_birds_paths, agent_id=agent_id, base_path=root_path)
        assert student_with_birds_ds is not None

        student_alone_ds.coords['setup'] = 'student_alone'
        student_with_birds_ds.coords['setup'] = 'student_with_birds'

        ds = xr.concat([student_alone_ds, student_with_birds_ds], dim='setup')

        ds = filter_agent_id(ds, agent_id=agent_id, drop=True)

        thermal_info_ds = resolve_thermal_info_from_bird_dataset2(
            bird_dataset_path, include_per_bird_data=False)
        ds = ds.merge(thermal_info_ds)

        cache_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_filepath, engine='scipy')
    else:
        print('loading dataset from cache file...')
        ds = xr.open_dataset(cache_filepath, decode_cf=True, engine='netcdf4')

    print('dataset loaded')

    ds = _preprocess(ds)
    return ds


def _preprocess(ds: xr.Dataset):

    ds = remove_thermal(ds, thermal_name='b0230')
    ds = remap_thermal_names(ds)

    return ds


def _resolve_dataset_paths(r_distances: list[int], experiment_setup: str,
                           policy_neptune_run_id: str, episode_count: int):

    base_path = Path(
        f'{experiment_setup}/{policy_neptune_run_id}/{episode_count}_episodes')

    def resolve_dataset(r: float):

        r_prime = r / np.sqrt(2)

        formatted_r = f'{r_prime:.8f}'.rstrip('0')

        return DatasetPath(
            id=f'{experiment_setup}_{r}m',
            start_distance_r_m=r,
            filepath=base_path /
            f'{experiment_setup}_{formatted_r}m_{formatted_r}m.csv')

    datasets = [resolve_dataset(r) for r in r_distances]

    return datasets


def _load_and_merge_datasets(
    datasets: list[DatasetPath],
    agent_id: str | None,
    base_path: Path = Path()) -> xr.Dataset | None:

    merged_ds: xr.Dataset | None = None

    for dataset in datasets:

        print(f'loading {dataset.filepath}...')
        loaded = _load_dataset(dataset, base_path=base_path)
        df = loaded.df

        agent_id_index: list[str] = []
        if agent_id is not None:
            df = filter_agent_id_pd(df, agent_id)
            agent_id_index = ['agent_id']

        df = df.set_index(['thermal', 'episode', *agent_id_index, 'time_s'])

        print('creating xarray dataset...')
        ds = xr.Dataset.from_dataframe(df)
        ds.coords['start_distance'] = loaded.info.start_distance_r_m
        ds.coords['start_distance'].attrs['units'] = 'meters'

        if merged_ds is None:
            merged_ds = ds
            continue

        print('concatenating datasets...')
        merged_ds = xr.concat([merged_ds, ds], dim='start_distance')

    return merged_ds


def _load_dataset(dataset: DatasetPath, base_path: Path):
    df = pd.read_csv(base_path / dataset.filepath)
    return LoadedDataset(df=df,
                         info=Info(
                             id=dataset.id,
                             start_distance_r_m=dataset.start_distance_r_m))
