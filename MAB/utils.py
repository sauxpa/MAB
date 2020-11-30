""" Packages import """
import numpy as np
from numba import jit
import bottleneck as bn
from scipy.special import softmax
import scipy.stats as sc


@jit(nopython=True)
def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


@jit(nopython=True)
def rd_choice(vec, size):
    """
    jit version of np.random.choice (slightly improve the computation time)
    """
    return np.random.choice(vec, size=size, replace=False)


@jit(nopython=True)
def hypergeom_sample(s1, n1, n2):
    """
    jit version of np.random.choice (slightly improve the computation time)
    """
    return np.random.hypergeometric(s1, n1 - s1, nsample=n2)


def rollavg_bottlneck(a, n):
    """
    :param a: array
    :param n: window of the rolling average
    :return: A fast function for computing moving averages
    """
    return bn.move_mean(a, window=n, min_count=n)


@jit(nopython=True)
def get_leader(Na, Sa, l_prev):
    """
    :param Na: Number of pulls of each arm (array)
    :param Sa: Sum of rewards of each arm (array)
    :param l_prev: previous leader
    :return: Leader for SSMC and SDA algorithms
    """
    m = np.amax(Na)
    n_argmax = np.nonzero(Na == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        s_max = Sa[n_argmax].max()
        s_argmax = np.nonzero(Sa[n_argmax] == s_max)[0]
        if np.nonzero(n_argmax[s_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(s_argmax)]


def get_leader_weighted(Na, Sa, visit_times, t, gamma, l_prev):
    """
    :param Na: Number of pulls of each arm (array)
    :param Sa: Sum of rewards of each arm (array)
    :param visit_times: time index of visits to each arm (array)
    :param t: current time (float)
    :param gamma: discount factor
    :param l_prev: previous leader
    :return: Leader for SSMC and SDA algorithms
    """
    nb_arms = len(Na)
    Na_weighted = np.zeros(nb_arms)
    for k in range(nb_arms):
        Na_weighted[k] = np.sum(gamma ** (t -  np.array(visit_times[k])))
        # Na_weighted[k] = Na[k]
    m = np.amax(Na_weighted)
    n_argmax = np.nonzero(Na_weighted == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        s_max = Sa[n_argmax].max()
        s_argmax = np.nonzero(Sa[n_argmax] == s_max)[0]
        if np.nonzero(n_argmax[s_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(s_argmax)]


def get_leader_ra(rewards_arm, empirical_risk_measure, Na, l_prev):
    """
    :param rewards_arm: rewards of each arm (array)
    :param empirical_risk_measure: function
    :param Na: Number of pulls of each arm (array)
    :param l_prev: previous leader
    :return: Leader for SSMC and SDA algorithms
    """
    m = np.amax(Na)
    n_argmax = np.nonzero(Na == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        rhos = empirical_risk_measure(np.array(rewards_arm)[n_argmax])
        rho_max = rhos.max()
        rho_argmax = np.nonzero(rhos == rho_max)[0]
        if np.nonzero(n_argmax[rho_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(rho_argmax)]


def get_leader_cvar(sorted_rewards_arm, idx_quantile, empirical_cvar, Na, l_prev):
    """
    :param sorted_rewards_arm: rewards of each arm (array), sorted
    :param idx_quantile: index of alpha quantile for each arm (array)
    :param empirical_cvar: function
    :param Na: Number of pulls of each arm (array)
    :param l_prev: previous leader
    :return: Leader for SSMC and SDA algorithms
    """
    m = np.amax(Na)
    n_argmax = np.nonzero(Na == m)[0]
    if n_argmax.shape[0] == 1:
        l = n_argmax[0]
        return l
    else:
        rhos = np.array(
                [
                    empirical_cvar(
                        np.array(sorted_rewards_arm)[n],
                        idx_quantile[n]
                        ) for n in n_argmax
                ]
            )
        rho_max = rhos.max()
        rho_argmax = np.nonzero(rhos == rho_max)[0]
        if np.nonzero(n_argmax[rho_argmax] == l_prev)[0].shape[0] > 0:
            return l_prev
    return n_argmax[np.random.choice(rho_argmax)]


def get_SSMC_star_min(rewards_l, n_challenger, reshape_size):
    """
    little helper for SSMC*
    """
    return (np.array(rewards_l)[:n_challenger * reshape_size].reshape(
        (reshape_size, n_challenger))).mean(axis=1).min()


def convert_tg_mean(mu, scale, step=1e-7):
    """
    :param mu: mean of the underlying gaussian r.v
    :param scale: scale of the underlying gaussian r.v
    :param step: precision of the numerical integration
    :return: compute the mean of the Truncated Gaussian r.v knowing the parameters of its
    associated Gaussian r.v
    """
    X = np.arange(0, 1, step)
    return (X * sc.norm.pdf(X, loc=mu, scale=scale)).mean() + 1 - sc.norm.cdf(1, loc=mu, scale=scale)
