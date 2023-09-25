from typing import Any, Union

import numpy as np

from bsn_jax.data.genz_gaussian import (
    Genz_continuous,
    uniform_to_gaussian,
    integral_Genz_continuous,
    Genz_cornerpeak,
    integral_Genz_cornerpeak,
    Genz_discontinuous,
    integral_Genz_discontinuous,
    Genz_gaussian,
    integral_Genz_gaussian,
    Genz_oscillatory,
    integral_Genz_oscillatory,
    Genz_productpeak,
    integral_Genz_productpeak,
)


def sample_standard_normal(n: int, dim: int):
    mean = np.zeros(dim)
    cov = np.identity(dim)
    x = np.random.multivariate_normal(mean, cov, n)
    return x


def score_function_standard_normal(x):
    return -x


class IntegrationDataset:
    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def f(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def true_integration_value(self) -> Union[None, float]:
        raise NotImplemented

    def sample(self, n: int) -> np.ndarray:
        raise NotImplemented

    def get_x_test(self) -> np.ndarray:
        raise NotImplemented

    def return_data_set(self, n: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        x = self.sample(n)
        score = self.score(x)
        y = self.f(x)
        x_test = self.get_x_test()
        score_test = self.score(x_test)
        return x, score, y, x_test, score_test


class GenzContinuousDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_continuous(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_continuous(self.a, self.u).item()


class GenzCornerpeakDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_cornerpeak(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_cornerpeak(self.a, self.u).item()


class GenzDiscontinuousDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_discontinuous(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_discontinuous(self.a, self.u).item()


class GenzGaussianDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_gaussian(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_gaussian(self.a, self.u).item()


class GenzOscillatoryDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_oscillatory(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_oscillatory(self.a, self.u).item()


class GenzProductpeakDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x: np.ndarray) -> np.ndarray:
        return score_function_standard_normal(x)

    def f_uniform(self, x: np.ndarray) -> Any:
        return Genz_productpeak(x, self.a, self.u)

    def f(self, x: np.ndarray) -> np.ndarray:
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, n: int) -> np.ndarray:
        return sample_standard_normal(n, self.dim)

    def get_x_test(self) -> np.ndarray:
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self) -> float:
        return integral_Genz_productpeak(self.a, self.u).item()
