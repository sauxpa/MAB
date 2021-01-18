""" Packages import """
from .MAB import *
import numpy as np


class CauchyMAB(GenericMAB):
    """
    Cauchy Bandit Problem
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
        super().__init__(methods=['C']*len(p), p=p)
        # Parameters used for stop learning policy
        self.best_arm = self.get_best_arm()

    def get_best_arm(self):
        ind = np.nonzero(self.means == np.amax(self.means))[0]
        var = [self.MAB[arm].variance for arm in ind]
        u = np.argmin(var)
        return ind[u]
