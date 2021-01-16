""" Packages import """
from .MAB import *
from .MAB_nonstationary import *
from .RAMAB import *
import numpy as np


class LogGaussianMAB(GenericMAB):
    """
    LogGaussian Bandit Problem
    """
    def __init__(self,
                 p,
                 risk_measure='mean'  # dummy
                 ):
        """
        Initialization
        :param p: np.array, true values of parameters for each arm
        """
        # Initialization of arms from GenericMAB
        super().__init__(methods=['LG']*len(p), p=p)
        # Parameters used for stop learning policy
        self.best_arm = self.get_best_arm()
        # Careful: Cp is the bound only with same variance for each arm
        self.Cp = sum([(self.mu_max - arm.mu) / self.kl2(arm.mu, self.mu_max, arm.eta, self.MAB[self.best_arm].eta)
                       for arm in self.MAB if arm.mu != self.mu_max])

    def get_best_arm(self):
        ind = np.nonzero(self.means == np.amax(self.means))[0]
        var = [self.MAB[arm].variance for arm in ind]
        u = np.argmin(var)
        return ind[u]

    @staticmethod
    def kl2(mu1, mu2, sigma1, sigma2):
        """
        Implementation of the Kullback-Leibler divergence for two Gaussian with different std
        :param x: float
        :param y: float
        :return: float, KL(B(x), B(y))
        """
        return np.log(sigma2/sigma1) + 0.5 * (sigma1**2/sigma2**2 + (mu2-mu1)**2/sigma2**2 - 1)
