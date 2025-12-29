import numpy as np


def create_uniform_polar_grid(r_max_m: float, r_step_m: float):

    # expected cell area
    A_cell_m2 = r_step_m**2

    # calculate N_theta for each ring
    n_r_edges = int(np.floor(r_max_m / r_step_m)) + 1
    r_edges = np.array([r_step_m * i for i in range(0, n_r_edges)])

    # calculate n_theta for each ring using A_ring / A_cell
    n_thetas = [
        int(np.round(
            (np.pi * (r_edges[i]**2 - r_edges[i - 1]**2)) / A_cell_m2))
        for i in range(1, n_r_edges)
    ]

    # create r<->theta pairs
    r_with_angles = None
    for i in range(0, len(n_thetas)):
        n_theta = n_thetas[i]
        angle = 2 * np.pi / n_theta
        angles = np.arange(0, 2 * np.pi, angle)

        r_outer = r_edges[i + 1]
        current_r_with_angles = np.stack(
            (np.full_like(angles, r_outer), angles), axis=1)
        r_with_angles = current_r_with_angles if r_with_angles is None else np.vstack(
            (r_with_angles, current_r_with_angles))

    assert (r_with_angles is not None)

    xs = r_with_angles[:, 0] * np.cos(r_with_angles[:, 1])
    ys = r_with_angles[:, 0] * np.sin(r_with_angles[:, 1])

    # add origo
    xs = np.concatenate(([0], xs))
    ys = np.concatenate(([0], ys))

    return xs, ys
