from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from ..common.thermal_map import remap_thermal_names
from ..common.dataset_utils import (filter_agent_id, filter_agent_id_pd,
                                    load_bird_dataset, remove_thermal,
                                    resolve_thermal_info_from_bird_dataset2)
import xarray as xr


@dataclass
class Info:
    id: str


@dataclass(kw_only=True)
class DatasetPath(Info):
    filepath: Path | str


def load_close_distance_dataset(policy_neptune_run_id: str,
                                episode_count: int):
    root_path = Path(
        'results/eval/realistic/peer_informed/from_close_distance_using_bird_wing_loadings'
    )

    bird_dataset_path = Path(
        'data/bird_comparison/processed/stork_trajectories_as_observation_log/merged_observation_log.csv'
    )

    cache_dir = Path('data/cache/from_close_distance_using_bird_wing_loadings')
    cache_filepath = cache_dir / 'from_close_distance_using_bird_wing_loadings.nc'

    if not cache_filepath.exists():

        print('loading ai datasets...')
        agent_id = 'student0'
        ds_list = [
            _load_dataset(dataset,
                          agent_id=agent_id,
                          drop_agent_id=True,
                          base_path=root_path) for dataset in
            _resolve_dataset_paths(policy_neptune_run_id=policy_neptune_run_id,
                                   episode_count=episode_count)
        ]

        # load bird dataset
        print('loading bird dataset...')
        bird_ds = load_bird_dataset(bird_dataset_path)
        bird_ds.coords['setup'] = 'birds'

        ds = xr.concat([*ds_list, bird_ds], dim='setup')

        # include thermal info
        bird_thermal_info_ds = resolve_thermal_info_from_bird_dataset2(
            bird_dataset_path, include_per_bird_data=True)
        ds = ds.merge(bird_thermal_info_ds)

        cache_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_filepath, engine='scipy')

    else:
        ds = xr.open_dataset(cache_filepath, decode_cf=True, engine='netcdf4')

    print('dataset loaded')

    ds = _preprocess(ds)
    return ds


def _preprocess(ds: xr.Dataset):

    ds = remove_thermal(ds, thermal_name='b0230')
    ds = remap_thermal_names(ds)

    return ds


def _resolve_dataset_paths(policy_neptune_run_id: str, episode_count: int):
    datasets: list[DatasetPath] = [
        DatasetPath(
            id='student_alone',
            filepath=
            f'student_alone/{policy_neptune_run_id}/{episode_count}_episodes/student_alone_using_bird_wing_loadings.csv'
        ),
        DatasetPath(
            id='student_with_birds',
            filepath=
            f'student_with_birds/{policy_neptune_run_id}/{episode_count}_episodes/student_with_birds_using_bird_wing_loadings.csv'
        )
    ]

    return datasets


def _load_dataset(dataset: DatasetPath, agent_id: str, drop_agent_id: bool,
                  base_path: Path) -> xr.Dataset:

    df = pd.read_csv(base_path / dataset.filepath)
    df = filter_agent_id_pd(df=df, agent_id=agent_id)

    df = df.set_index(
        ['thermal', 'bird_name', 'episode', 'agent_id', 'time_s'])
    ds = xr.Dataset.from_dataframe(df)

    ds = filter_agent_id(ds, agent_id=agent_id, drop=drop_agent_id)

    ds.coords['setup'] = dataset.id

    return ds
