from typing import Match
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib
import matplotlib.figure
import sigfig
import re


def remove_thermal(ds: xr.Dataset, thermal_name: str):
    mask = ds.thermal != thermal_name
    ds = ds.isel(thermal=mask)

    return ds


def filter_agent_id_pd(df: pd.DataFrame, agent_id: str):
    filtered_df = df[df['agent_id'] == agent_id]
    assert isinstance(filtered_df, pd.DataFrame)
    return filtered_df


def filter_agent_id(ds: xr.Dataset, agent_id: str, drop: bool) -> xr.Dataset:
    ds = ds.sel(agent_id=agent_id, drop=drop)
    return ds


def load_bird_dataset(path: Path) -> xr.Dataset:

    df = pd.read_csv(path)
    df = df.set_index(['thermal', 'episode', 'bird_name', 'time_s'])

    ds = xr.Dataset.from_dataframe(df)

    return ds


def resolve_thermal_info_from_bird_dataset2(bird_dataset_path: Path,
                                            include_per_bird_data: bool):

    bird_ds = load_bird_dataset(bird_dataset_path)

    # altitudes
    thermal_bird_min_altitude_per_bird_da = bird_ds['position_earth_m_z'].min(
        dim=['episode', 'time_s']).assign_attrs(units='meters')
    thermal_bird_max_altitude_per_bird_da = bird_ds['position_earth_m_z'].max(
        dim=['episode', 'time_s']).assign_attrs(units='meters')

    thermal_bird_mean_min_altitude_da = thermal_bird_min_altitude_per_bird_da.mean(
        dim=['bird_name']).assign_attrs(units='meters')
    thermal_bird_mean_max_altitude_da = thermal_bird_max_altitude_per_bird_da.mean(
        dim=['bird_name']).assign_attrs(units='meters')

    thermal_bird_min_altitude_da = thermal_bird_min_altitude_per_bird_da.min(
        dim=['bird_name']).assign_attrs(units='meters')
    thermal_bird_max_altitude_da = thermal_bird_max_altitude_per_bird_da.max(
        dim=['bird_name']).assign_attrs(units='meters')

    thermal_info_ds = xr.Dataset({
        'thermal_bird_mean_minimum_altitude':
        thermal_bird_mean_min_altitude_da,
        'thermal_bird_mean_maximum_altitude':
        thermal_bird_mean_max_altitude_da,
        'thermal_bird_minimum_altitude':
        thermal_bird_min_altitude_da,
        'thermal_bird_maximum_altitude':
        thermal_bird_max_altitude_da
    })

    if include_per_bird_data:

        thermal_info_ds = {
            **thermal_info_ds, 'thermal_bird_minimum_altitude_per_bird':
            thermal_bird_min_altitude_per_bird_da,
            'thermal_bird_maximum_altitude_per_bird':
            thermal_bird_max_altitude_per_bird_da
        }

    return thermal_info_ds


def save_figure(fig: matplotlib.figure.Figure,
                output_dir: Path,
                name: str,
                transparent_svg: bool = True,
                png: bool = True,
                svg: bool = True):

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    backend = matplotlib.get_backend()

    if png:
        png_path = output_dir / f'{name}.png'
        print(f'saving {png_path} using "{backend}" backend...')
        fig.savefig(png_path)

    if svg:
        svg_path = output_dir / f'{name}.svg'
        print(f'saving {svg_path} using "{backend}" backend...')

        original_svg_path = None
        if backend == 'cairo':
            # cairo does not write pt in the svg, fix it
            original_svg_path = svg_path
            svg_path = output_dir / f'{name}_cairo.svg'

        fig.savefig(svg_path, transparent=transparent_svg)

        if backend == 'cairo':
            assert original_svg_path is not None
            _fix_cairo_svg(cairo_svg_path=svg_path,
                           output_svg_path=original_svg_path)


def _fix_cairo_svg(cairo_svg_path: Path, output_svg_path: Path):
    with open(cairo_svg_path, 'r', encoding='utf-8') as f:
        svg = f.read()

    def add_pt(match: Match):
        return match.group(
            1) + f'width="{match.group(2)}pt" height="{match.group(3)}pt"'

    # include pt in width and height at the root
    svg = re.sub(r'(\<svg.*)width="([\d.]+)" height="([\d.]+)"',
                 add_pt,
                 svg,
                 count=1)

    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)


def write_dataframe(df: pd.DataFrame,
                    output_dir: Path,
                    name: str,
                    float_format='%.2f',
                    longtable=False):

    output_dir.mkdir(exist_ok=True, parents=True)

    df.to_excel(output_dir / f'{name}.xlsx', float_format=float_format)

    latex_str = df.to_latex(index=True, escape=False, longtable=longtable)

    latex_str = latex_str.replace('\\toprule', '\\hline')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline')

    with open(output_dir / f'{name}.tex', 'w') as f:
        f.write(latex_str)


def format_p_star(p):

    if p <= 0.0001:
        return '****'
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return sigfig.round(p, sigfigs=3)


def calculate_standard_error_of_mean(std_da: xr.DataArray,
                                     n_for_std_da: xr.DataArray):
    # https://resources.nu.edu/statsresources/ComputingSEM
    sem = std_da / np.sqrt(n_for_std_da)
    return sem


def calculate_95_confidence_interval_from_sem(sem_da: xr.DataArray):
    ci95 = sem_da * 1.960

    return ci95
