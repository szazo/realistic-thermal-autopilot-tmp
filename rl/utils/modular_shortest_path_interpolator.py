import numpy as np
import numpy.typing as npt


class ModularShortestPathInterpolator():
    _x: np.ndarray
    _y: np.ndarray

    _low: float
    _high: float

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, low: float,
                 high: float):

        self._x = np.array(x)
        self._y = np.array(y)
        self._low = low
        self._high = high

    def __call__(self, xi: float):

        if xi < self._x[0] or xi > self._x[-1]:
            return None

        # the left index will be the insertion point minus 1
        idx = np.searchsorted(self._x, xi, side='left') - 1
        # however we need to handle the boundary cases
        idx = np.clip(idx, 0, len(self._x) - 2)

        x0 = self._x[idx]
        x1 = self._x[idx + 1]

        y0 = self._y[idx]
        y1 = self._y[idx + 1]

        t = (xi - x0) / (x1 - x0)

        # support for e.g. [-pi,pi]
        mod = self._high - self._low

        # rescale
        y0_prime = (y0 - self._low) % mod
        y1_prime = (y1 - self._low) % mod

        # modular shortest difference
        diff = (y1_prime - y0_prime) % mod
        diff = np.where(diff > (mod / 2), diff - mod, diff)

        # interpolate
        result = (y0_prime + t * diff) % mod

        # transform back to the original interval
        result += self._low

        return result
