""" Packages import """
import numpy as np


class AbstractNonStationaryArm(object):
    def __init__(self,
                 mean,
                 variance,
                 random_state=0,
                 ):
        """
        :param mean: function, time-varying expectation of the arm
        :param variance: function, time-varying variance of the arm
        :param random_state: int, seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance
        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass


class ArmGaussianMA(AbstractNonStationaryArm):
    def __init__(self, mu, eta, alpha=None, random_state=0):
        """
        :param mu: function, time-varying mean parameter in gaussian distribution
        :param eta: float, std parameter in gaussian distribution
        :param alpha: float or None, risk aversion level
        :param random_state: int, seed to make experiments reproducible
        """
        self.mu = mu
        self.eta = eta

        super(ArmGaussianMA, self).__init__(mean=mu,
                                            variance=lambda t: eta**2,
                                            random_state=random_state,
                                            )

    def sample(self, t, N=1):
        """
        Sampling strategy
        :param t: int, sample time
        :param N: int, sample size
        :return: float, a sample from the arm
        """
        return self.local_random.normal(self.mu(t), self.eta, N)
