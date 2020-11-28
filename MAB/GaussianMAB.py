""" Packages import """
from .MAB import *
from .RAMAB import *
import numpy as np


class GaussianMAB(GenericMAB):
    """
    Gaussian Bandit Problem
    """
    def __init__(self,
                 p,
                 risk_measure='mean'  # dummy
                 ):
        """
        Initialization
        :param p: np.array, true values of 1/lambda for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['G']*len(p), p=p)
        # Parameters used for stop learning policy
        self.best_arm = self.get_best_arm()
        # Careful: Cp is the bound only with same variance for each arm
        self.Cp = sum([(self.mu_max - arm.mu) / self.kl2(arm.mu, self.mu_max, arm.eta, self.MAB[self.best_arm].eta)
                       for arm in self.MAB if arm.mu != self.mu_max])

    def get_best_arm(self):
        ind = np.nonzero(self.means == np.amax(self.means))[0]
        std = [self.MAB[arm].eta for arm in ind]
        u = np.argmin(std)
        return ind[u]

    @staticmethod
    def kl(mu1, mu2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian N(mu, 1)
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return (mu2-mu1)**2/2

    @staticmethod
    def kl2(mu1, mu2, sigma1, sigma2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian with different std
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return np.log(sigma2/sigma1) + 0.5 * (sigma1**2/sigma2**2 + (mu2-mu1)**2/sigma2**2 - 1)

    def TS(self, T):
        """
        Thompson Sampling for Gaussian distributions with known variance, and an inproper uniform prior
        on the mean
        :param T: Time Horizon
        :return: Tracker2 object
        """
        eta = np.array([arm.eta for arm in self.MAB])

        def f(x):
            return np.random.normal(x.Sa/x.Na, eta/np.sqrt(x.Na))
        return self.Index_Policy(T, f)

    def kl_ucb(self, T, f):
        """
        Implementation of KL-UCB for Gaussian bandits
        :param T: Time Horizon
        :param rho: coefficient for the upper bound
        :return:
        """
        def index_func(x):
            return x.Sa / x.Na + np.sqrt(f(x.t)*2 / x.Na)
        return self.Index_Policy(T, index_func)


class GaussianRAMAB(GenericRAMAB):
    """
    Gaussian Risk Averse Bandit Problem
    """
    def __init__(self,
                 p,
                 risk_measure,
                 ):
        """
        Initialization
        :param p: np.array, true values of 1/lambda for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['G'] * len(p),
                         p=p,
                         risk_measure=risk_measure,
                         )
        # Parameters used for stop learning policy
        self.best_arm = self.get_best_arm(risk_measure)

    def get_best_arm(self, risk_measure):
        if risk_measure == 'mean':
            ind = np.nonzero(self.means == np.amax(self.means))[0]
            std = [self.MAB[arm].eta for arm in ind]
            u = np.argmin(std)
        else:
            rhos = self.risk_measures[self.risk_measure]
            ind = np.nonzero(
                rhos == np.amax(rhos)
                )[0]
            mus = [self.MAB[arm].mu for arm in ind]
            u = np.argmax(mus)  # in case of ties
        return ind[u]

    @staticmethod
    def kl(mu1, mu2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian N(mu, 1)
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return (mu2 - mu1) ** 2 / 2

    @staticmethod
    def kl2(mu1, mu2, sigma1, sigma2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian with different std
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return np.log(sigma2 / sigma1) + 0.5 * (
            sigma1 ** 2 / sigma2 ** 2 + (mu2 - mu1) ** 2 / sigma2 ** 2 - 1
            )

    @property
    def empirical_erm(self):
        """
        Implementation of empirical estimator for the Entropic Risk Measure.
        :param samples: list or np.ndarray
        """
        return lambda samples: np.mean(samples) - 0.5 * self.alpha * np.var(samples)
