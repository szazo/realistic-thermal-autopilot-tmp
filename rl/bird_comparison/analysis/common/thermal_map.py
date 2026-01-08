import xarray as xr


def thermal_colormap():

    return {
        'R0': '#00b845ff',
        'R1': '#0a5ca3ff',
        'R2': '#ff9400ff',
        'R3': '#ff2b00ff',
        'R4': '#825997ff',
        'R5': '#454545ff'
    }


def resolve_thermal_mapping():
    thermal_mapping = {
        'b0230': 'R0',
        'b010': 'R1',
        'b072': 'R2',
        'b077': 'R3',
        'b112': 'R4',
        'b121': 'R5'
    }
    return thermal_mapping


def remap_thermal_names(ds: xr.Dataset):
    # rename
    thermal_mapping = resolve_thermal_mapping()

    # create a new coordinate for the existing dimension 'thermal'

    # copy the old as id
    ds = ds.assign_coords(thermal_id=ds.coords['thermal'])

    # use the new names
    ds = ds.assign_coords(
        thermal=[f'{thermal_mapping[v]}' for v in ds['thermal'].values])

    return ds
