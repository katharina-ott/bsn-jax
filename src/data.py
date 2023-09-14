import numpy as np

from genz_gaussian import Genz_continuous, integral_Genz_continuous, uniform_to_gaussian, Genz_cornerpeak, \
    integral_Genz_cornerpeak, Genz_discontinuous, integral_Genz_discontinuous, Genz_gaussian, integral_Genz_gaussian, \
    Genz_oscillatory, integral_Genz_oscillatory, Genz_productpeak, integral_Genz_productpeak


def sample_standard_normal(N: int, dim: int):
    mean = np.zeros(dim)
    cov = np.identity(dim)
    x = np.random.multivariate_normal(mean, cov, N)
    return x


def score_function_standard_normal(x):
    return -x


class IntegrationDataset():
    def score(self, x):
        raise NotImplemented

    def f(self, x):
        raise NotImplemented

    def true_integration_value(self):
        raise NotImplemented

    def sample(self, N: int):
        raise NotImplemented

    def get_x_test(self):
        raise NotImplemented

    def return_data_set(self, N):
        x = self.sample(N)
        score = self.score(x)
        y = self.f(x)
        x_test = self.get_x_test()
        score_test = self.score(x_test)
        return x, score, y, x_test, score_test


class GenzContinuousDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_continuous(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_continuous(self.a, self.u)


class GenzCornerpeakDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_cornerpeak(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_cornerpeak(self.a, self.u)


class GenzDiscontinuousDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_discontinuous(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_discontinuous(self.a, self.u)


class GenzGaussianDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_gaussian(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_gaussian(self.a, self.u)


class GenzOscillatoryDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_oscillatory(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_oscillatory(self.a, self.u)


class GenzProductpeakDataSet1D(IntegrationDataset):
    dim = 1
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)

    def score(self, x):
        return score_function_standard_normal(x)

    def f_uniform(self, x):
        return Genz_productpeak(x, self.a, self.u)

    def f(self, x):
        return uniform_to_gaussian(self.f_uniform)(x)

    def sample(self, N: int):
        return sample_standard_normal(N, self.dim)

    def get_x_test(self):
        return np.linspace(-5, 5, 200)[:, None]

    def true_integration_value(self):
        return integral_Genz_productpeak(self.a, self.u)
