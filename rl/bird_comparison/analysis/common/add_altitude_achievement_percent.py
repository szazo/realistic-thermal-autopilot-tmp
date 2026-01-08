import xarray as xr


def add_altitude_achievement_percent(
        ds: xr.Dataset, bird_maximum_altitude_reference_variable: str):

    altitude_achievement_percent_da = calculate_achievement_percent(
        agent_initial_altitude_da=ds['agent_initial_altitude'],
        agent_maximum_altitude_da=ds['agent_maximum_altitude'],
        thermal_bird_maximum_altitude_da=ds[
            bird_maximum_altitude_reference_variable])
    ds = ds.assign(
        altitude_achievement_percent=altitude_achievement_percent_da)

    return ds


def add_agent_initial_and_maximum_altitude(ds: xr.Dataset):

    agent_initial_altitude_da = agent_initial_altitude(ds)
    agent_maximum_altitude_da = agent_maximum_altitude(ds)

    ds = ds.assign(agent_initial_altitude=agent_initial_altitude_da)
    ds = ds.assign(agent_maximum_altitude=agent_maximum_altitude_da)

    return ds


def agent_initial_altitude(ds: xr.Dataset) -> xr.DataArray:
    # search first non-nan
    altitude_da = ds['position_earth_m_z']
    mask = ~altitude_da.isnull()
    first_idx = mask.argmax(dim='time_s')

    agent_initial_altitude_da = altitude_da.isel(time_s=first_idx)

    return agent_initial_altitude_da


def agent_maximum_altitude(ds: xr.Dataset) -> xr.DataArray:

    agent_maximum_altitude_da = ds['position_earth_m_z'].max(
        dim=['time_s']).assign_attrs(units='meters')

    return agent_maximum_altitude_da


def calculate_achievement_percent(
        agent_initial_altitude_da: xr.DataArray,
        agent_maximum_altitude_da: xr.DataArray,
        thermal_bird_maximum_altitude_da: xr.DataArray):

    altitude_achievement_percent_da = (
        agent_maximum_altitude_da - agent_initial_altitude_da) / (
            thermal_bird_maximum_altitude_da - agent_initial_altitude_da) * 100
    altitude_achievement_percent_da = altitude_achievement_percent_da.clip(
        min=0., max=100.)

    return altitude_achievement_percent_da
