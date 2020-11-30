""" Packages import """
import numpy as np
from scipy.stats import truncnorm as trunc_norm, norm
from .utils import convert_tg_mean


class AbstractArm(object):
    def __init__(self,
                 mean,
                 variance,
                 alpha=None,
                 random_state=0,
                 ):
        """
        :param mean: float, expectation of the arm
        :param variance: float, variance of the arm
        :param alpha: float or None, risk aversion level
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance
        self.alpha = alpha
        self.local_random = np.random.RandomState(random_state)
        self.risk_measures = {
            'erm': self._get_erm(),
            'cvar': self._get_cvar(),
        }

    def sample(self):
        pass

    @property
    def erm(self):
        return self.risk_measures.get('erm', None)

    @property
    def cvar(self):
        return self.risk_measures.get('cvar', None)

    def _get_cvar(self, eps=1e-3):
        """cVaR calculation on a discrete grid of size cvar_epsilon.
        """
        try:
            quantile_grid = np.arange(eps, 1 - eps, eps)
            quantiles = self.ppf(quantile_grid)
            # cVar is the conditional expectation given a level of VaR.
            return quantiles[quantile_grid < self.alpha].mean()
        except AttributeError:
            raise AttributeError('Method ppf not implemented')


class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param p: float, mean parameter
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)

    def sample(self, N=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return (self.local_random.rand(N) < self.p)*1.


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        :param a: int, alpha coefficient in beta distribution
        :param b: int, beta coefficient in beta distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a/(a + b),
                                      variance=(a * b)/((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state)

    def sample(self, N=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.beta(self.a, self.b, N)


class ArmGaussian(AbstractArm):
    def __init__(self, mu, eta, alpha=None, random_state=0):
        """
        :param mu: float, mean parameter in gaussian distribution
        :param eta: float, std parameter in gaussian distribution
        :param alpha: float or None, risk aversion level
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta

        super(ArmGaussian, self).__init__(mean=mu,
                                          variance=eta**2,
                                          alpha=alpha,
                                          random_state=random_state,
                                          )

    def sample(self, N=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.normal(self.mu, self.eta, N)

    def _get_erm(self):
        return self.mu - self.alpha * self.eta ** 2 / 2

    def ppf(self, q):
        """
        Percentile function (inverse cumulative distribution function)
        :param q: np.ndarray, quantiles to evaluate
        :return: np.ndarray, quantiles
        """
        return norm.ppf(q, self.mu, self.eta)


class ArmFinite(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        :param X: np.array, support of the distribution
        :param P: np.array, associated probabilities
        :param random_state: int, seed to make experiments reproducible
        """
        self.X = X
        self.P = P
        mean = np.sum(X * P)
        super(ArmFinite, self).__init__(mean=mean,
                                        variance=np.sum(X ** 2 * P) - mean ** 2,
                                        random_state=random_state)

    def sample(self, N=1):
        """
        Sampling strategy for an arm with a finite support and the associated probability distribution
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        i = self.local_random.choice(len(self.P), size=N, p=self.P)
        reward = self.X[i]
        return reward


class ArmExponential(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        :param mu: float, mean parameter in gaussian distribution
        :param eta: float, std parameter in gaussian distribution
        :param random_state: int, seed to make experiments reproducible
        """
        self.p = p
        super(ArmExponential, self).__init__(mean=p,
                                             variance=p**2,
                                             random_state=random_state
                                             )

    def sample(self, N=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.exponential(self.p, N)


class dirac():
    def __init__(self, c, random_state):
        """
        :param mean: float, expectation of the arm
        :param variance: float, variance of the arm
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = c
        self.variance = 0
        self.local_random = np.random.RandomState(random_state)

    def sample(self, N=1):
        return [self.mean,] * N


class ArmTG(AbstractArm):
    def __init__(self, mu, scale, random_state=0):
        """
        :param mu: mean
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.scale = scale
        self.dist = trunc_norm(-mu/scale, b=(1-mu)/scale, loc=mu, scale=scale)
        self.dist.random_state = random_state
        super(ArmTG, self).__init__(mean=convert_tg_mean(mu, scale),
                                    variance=scale**2,
                                    random_state=random_state
                                    )

    def sample(self, N=1):
        """
        Sampling strategy
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        x = self.local_random.normal(self.mu, self.scale, N)
        return x * (x > 0) * (x < 1) + (x > 1)
