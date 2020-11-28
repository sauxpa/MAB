""" Packages import """
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
import MAB.arms as arms
from tqdm import tqdm
from .utils import rd_argmax, get_leader_ra
from .tracker import Tracker2
from .utils import get_SSMC_star_min

mapping = {
    'B': arms.ArmBernoulli, 'beta': arms.ArmBeta, 'F': arms.ArmFinite,
    'G': arms.ArmGaussian, 'Exp': arms.ArmExponential, 'dirac': arms.dirac,
    'TG': arms.ArmTG
    }


def default_exp(x):
    """
    :param x: float
    :return: default exploration function for SDA algorithms
    """
    return 0
    # return np.sqrt(np.log(x))


class GenericRAMAB:
    """
    Generic class to simulate a Risk Averse Multi-Arm Bandit problem
    """
    def __init__(self,
                 methods,
                 p,
                 risk_measure='mean',
                 ):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution
            of each arm
        """
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.risk_measure = risk_measure

        self.means = np.array([el.mean for el in self.MAB])
        self.mu_max = np.max(self.means)

        # Risk aversion parameter
        alphas = np.unique([el.alpha for el in self.MAB])
        if len(alphas) > 1:
            raise Exception('Risk aversion parameter should be the same across arms')
        else:
            self.alpha = alphas[0]

        # Entropic Risk Measures (-1/alpha * log(E[exp(-alpha*X)]))
        erms = np.array([el.erm for el in self.MAB])
        erm_max = np.max(erms)

        self.risk_measures = {
            'erm': erms,
        }

        self.risk_measures_max = {
            'erm': erm_max,
        }

        self.mc_regret = None

    @staticmethod
    def generate_arms(methods, p):
        """
        Method for generating different arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(methods):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            alg = mapping[m]
            arms_list.append(alg(*args))
        return arms_list

    @staticmethod
    def kl(x, y):
        return None

    def MC_regret(self, method, N, T, param_dic, store_step=-1, regret_mode='mean'):
        """
        Average Regret on a Number of Experiments
        :param method: string, method used (UCB, Thompson Sampling, etc..)
        :param N: int, number of independent experiments
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods
        :param regret_mode: string, risk measure used for regret
        """
        mc_regret = np.zeros(T)
        store = store_step > 0
        if store:
            all_regret = np.zeros((np.arange(T)[::store_step].shape[0], N))
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            tr = alg(T, **param_dic)
            regret = tr.regret(regret_mode)
            mc_regret += regret
            if store:
                all_regret[:, i] = regret[::store_step]
        if store:
            return mc_regret / N, all_regret
        return mc_regret / N

    def DummyPolicy(self, T):
        """
        Implementation of a random policy consisting in randomly choosing one of the available arms. Only useful
        for checking that the behavior of the different policies is normal
        :param T:  int, time horizon
        :return: means, arm sequence
        """
        tr = Tracker2(
            self.means,
            T,
            alpha=self.alpha,
            risk_measure=self.risk_measure,
            risk_measures=self.risk_measures,
            )
        tr.arm_sequence = np.random.randint(self.nb_arms, size=T)
        return tr

    def ExploreCommit(self, T, m):
        """
        Implementation of Explore-then-Commit algorithm
        :param T: int, time horizon
        :param m: int, number of rounds before choosing the best action
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(
            self.means,
            T,
            alpha=self.alpha,
            risk_measure=self.risk_measure,
            risk_measures=self.risk_measures,
            )
        for t in range(m * self.nb_arms):
            arm = t % self.nb_arms
            tr.update(t, arm, self.MAB[arm].sample()[0])
        arm = rd_argmax(tr.Sa / tr.Na)
        for t in range(m * self.nb_arms, T):
            tr.update(t, arm, self.MAB[arm].sample()[0])
        return tr

    def Index_Policy(self, T, index_func, start_explo=1, store_rewards_arm=False):
        """
        Implementation of UCB1 algorithm
        :param T: int, time horizon
        :param start_explo: number of time to explore each arm before comparing index
        :param index_func: function which computes the index with the tracker
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(
            self.means,
            T,
            alpha=self.alpha,
            risk_measure=self.risk_measure,
            risk_measures=self.risk_measures,
            store_rewards_arm=store_rewards_arm,
            )
        for t in range(T):
            if t < self.nb_arms*start_explo:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(tr))
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def erm_ucb(self, T, f, verbose=False):
        """
        Implementation of RA-UCB
        (https://hal.archives-ouvertes.fr/hal-00821670/)
        which corresponds to the entropic risk measure.

        ---------
        Sometimes works...
        ---------

        :param T: Time Horizon
        :param f: coefficient for the upper bound
        :return:
        """
        if self.risk_measure != 'erm':
            raise ValueError('This RA-UCB is only compatible with entropic risk measure')

        def index_func(x):
            res = []
            for k in range(self.nb_arms):
                def K(x, r):
                    def expect_log(gamma):
                        return np.mean(
                            np.log(
                                1 - gamma / x.alpha * (
                                    1 - np.exp(
                                        -x.alpha * (
                                            np.array(x.rewards_arm[k]) - r
                                            )
                                        )
                                    )
                                )
                            )
                    res = minimize_scalar(
                        expect_log,
                        bounds=(0, x.alpha),
                        method='bounded',
                        # tol=1e-6,
                        )
                    if res.success:
                        return res.x

                def K_shift(r):
                    ret = K(x, r) - f(1 + x.t) / x.Na[k]
                    return ret

                ret = root_scalar(
                    K_shift,
                    x0=0.5 * x.risk_measures['erm'][k],
                    x1=2.0 * x.risk_measures['erm'][k],
                    method='secant',
                    rtol=1e-3,
                    )
                res.append(ret.root)

            return np.array(res)

        return self.Index_Policy(T, index_func, store_rewards_arm=True)

    @property
    def empirical_risk_measure(self):
        if self.risk_measure == 'erm':
            return self.empirical_erm

    def empirical_erm(self):
        pass

    def RB_SDA(self, T, explo_func=default_exp):
        """
        Implementation of RB-SDA
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
        """
        tr = Tracker2(
            self.means,
            T,
            alpha=self.alpha,
            risk_measure=self.risk_measure,
            risk_measures=self.risk_measures,
            store_rewards_arm=True,
            )
        r, t, l = 1, 0, -1
        empirical_risk_measure = self.empirical_risk_measure
        while t < self.nb_arms:
            arm = t
            tr.update(t, arm, self.MAB[arm].sample()[0])
            t += 1
        while t < T:
            l_prev = l
            # l = get_leader(tr.Na, tr.Sa, l_prev)
            l = get_leader_ra(tr.rewards_arm, empirical_risk_measure, tr.Na, l_prev)

            t_prev, forced_explo = t, explo_func(r)
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    tj = np.random.randint(tr.Na[l] - tr.Na[j])
                    lead_risk_measure = empirical_risk_measure(
                        tr.rewards_arm[l][tj:tj + int(tr.Na[j])],
                        )
                    challenger_risk_measure = empirical_risk_measure(
                        tr.rewards_arm[j],
                        )
                    # print('leader: {}'.format(lead_risk_measure))
                    # print('challenger: {}'.format(challenger_risk_measure))
                    if challenger_risk_measure >= lead_risk_measure and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                tr.update(t, l, self.MAB[l].sample()[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        tr.update(t, i, self.MAB[i].sample()[0])
                        t += 1
            r += 1
        return tr
