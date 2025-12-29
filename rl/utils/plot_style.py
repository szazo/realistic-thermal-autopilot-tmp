import scienceplots
from typing import Literal
import matplotlib
import matplotlib.pyplot as plt
import contextlib


def select_plot_style(style: Literal['default', 'science'],
                      additional_rc_params: dict = {},
                      engine_override: str | None = None):

    if style == 'default':
        return default_plot_style(additional_rc_params=additional_rc_params)
    elif style == 'science':
        return science_plot_style(additional_rc_params=additional_rc_params,
                                  engine_override=engine_override)
    else:
        raise Exception(f'invalid style: {style}')


@contextlib.contextmanager
def default_plot_style(additional_rc_params: dict = {}):

    additional_rc_params = {
        'font.family': 'sans-serif',
        'font.size': 7.0,
        'svg.fonttype': 'path',
        **additional_rc_params
    }

    with plt.style.context(['science', additional_rc_params]):

        yield


@contextlib.contextmanager
def science_plot_style(additional_rc_params: dict = {},
                       figsize_multiplier=1.,
                       engine_override: str | None = None):

    additional_rc_params = {
        'axes.titlesize': figsize_multiplier * 7,
        'axes.labelsize': figsize_multiplier * 8,
        'xtick.labelsize': figsize_multiplier * 6,
        'ytick.labelsize': figsize_multiplier * 6,
        'legend.fontsize': figsize_multiplier * 8,
        'axes.linewidth': figsize_multiplier * 0.28,
        'lines.linewidth': figsize_multiplier * 0.28,
        'grid.linewidth': 0.28 / 2,
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Helvetica',
        'figure.dpi': 320,
        **additional_rc_params
    }

    original_backend = matplotlib.get_backend()
    try:
        if engine_override is None:
            plt.switch_backend('Cairo')
        else:
            plt.switch_backend(engine_override)

        with plt.style.context(['science', additional_rc_params]):
            yield
    finally:
        plt.switch_backend(original_backend)
