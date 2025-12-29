from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.stats import truncnorm, skewnorm


@dataclass
class DistributionConfigBase:
    _target_: str = 'missing'


class Distribution(ABC):

    @abstractmethod
    def sample(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        pass


@dataclass(kw_only=True)
class NormalDistributionParams:

    sigma: float
    mean: float = 0.


@dataclass(kw_only=True)
class NormalDistributionConfig(NormalDistributionParams,
                               DistributionConfigBase):

    _target_: str = 'distributions.NormalDistribution'


class NormalDistribution(Distribution):

    _mean: float
    _sigma: float

    def __init__(self, mean: float, sigma: float):
        self._mean = mean
        self._sigma = sigma

    def sample(self, size: int | tuple[int, ...] = 1):
        return np.random.normal(loc=self._mean, scale=self._sigma, size=size)


@dataclass(kw_only=True)
class TruncatedNormalDistributionParams(NormalDistributionParams):
    min: float
    max: float


class TruncatedNormalDistribution(Distribution):

    _mean: float
    _sigma: float
    _min: float
    _max: float

    def __init__(self, mean: float, sigma: float, min: float, max: float):

        self._mean = mean
        self._sigma = sigma
        self._min = min
        self._max = max

    def sample(self, size: int | tuple[int, ...] = 1):

        a, b = (self._min - self._mean) / self._sigma, (
            self._max - self._mean) / self._sigma

        result = truncnorm.rvs(a,
                               b,
                               loc=self._mean,
                               scale=self._sigma,
                               size=size)
        assert isinstance(result, np.ndarray)

        return result


class SkewedNormalDistributionParams(NormalDistributionParams):
    alpha: float


class SkewedNormalDistribution(Distribution):

    _mean: float
    _sigma: float
    _alpha: float

    def __init__(self, mean: float, sigma: float, alpha: float):

        self._mean = mean
        self._sigma = sigma
        self._alpha = alpha

    def sample(self, size: int | tuple[int, ...] = 1):

        result = skewnorm.rvs(a=self._alpha,
                              loc=self._mean,
                              scale=self._sigma,
                              size=size)
        assert isinstance(result, np.ndarray)

        return result


@dataclass
class UniformDistributionParams():
    min: float
    max: float


class UniformDistribution(Distribution):

    _closed_low: float
    _open_high: float

    def __init__(self, min: float, max: float):

        self._closed_low = min
        self._open_high = max

    def sample(self, size: int | tuple[int, ...] = 1):

        result = np.random.uniform(low=self._closed_low,
                                   high=self._open_high,
                                   size=size)
        return result
