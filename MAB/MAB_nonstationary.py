""" Packages import """
import numpy as np
from scipy.special import softmax
import MAB.arms_nonstationary as arms
from tqdm import tqdm
from .utils import rd_argmax, rd_choice, rollavg_bottlneck, get_leader_weighted
from .tracker import Tracker2
from .utils import get_SSMC_star_min
# import sobol_seq  # for LDS-SDA

mapping = {
    'MAG': arms.ArmGaussianMA,
    }


def default_exp(x):
    """
    :param x: float
    :return: default exploration function for SDA algorithms
    """
    return 0
    # return np.sqrt(np.log(x))


class GenericNonStationaryMAB:
    """
    Generic class to simulate a Multi-Arm Bandit problem
    """
    def __init__(self, methods, p):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        """
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.means = np.array([el.mean for el in self.MAB])
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

    def MC_regret(self, method, N, T, param_dic, store_step=-1, risk_measure='mean'):
        """
        Average Regret on a Number of Experiments
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent experiments
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods
        :param risk_measure: str, dummy (for compatibility with RAMAB)
        """
        mc_regret = np.zeros(T)
        store = store_step > 0
        if store:
            all_regret = np.zeros((np.arange(T)[::store_step].shape[0], N))
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            tr = alg(T, **param_dic)
            regret = tr.regret_dynamic()
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
        tr = Tracker2(self.means, T)
        tr.arm_sequence = np.random.randint(self.nb_arms, size=T)
        return tr

    def ExploreCommit(self, T, m):
        """
        Implementation of Explore-then-Commit algorithm
        :param T: int, time horizon
        :param m: int, number of rounds before choosing the best action
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(self.means, T)
        for t in range(m * self.nb_arms):
            arm = t % self.nb_arms
            tr.update(t, arm, self.MAB[arm].sample(t)[0])
        arm = rd_argmax(tr.Sa / tr.Na)
        for t in range(m * self.nb_arms, T):
            tr.update(t, arm, self.MAB[arm].sample(t)[0])
        return tr

    def Index_Policy(self, T, index_func, start_explo=1, store_rewards_arm=False):
        """
        Implementation of UCB1 algorithm
        :param T: int, time horizon
        :param start_explo: number of time to explore each arm before comparing index
        :param index_func: function which computes the index with the tracker
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        tr = Tracker2(self.means, T, store_rewards_arm)
        for t in range(T):
            if t < self.nb_arms*start_explo:
                arm = t % self.nb_arms
            else:
                arm = rd_argmax(index_func(tr))
            reward = self.MAB[arm].sample(t)[0]
            tr.update(t, arm, reward)
        return tr

    def UCB1(self, T, rho=1.):
        """
        :param T: Time Horizon
        :param rho: coefficient for the upper bound
        :return:
        """
        def index_func(x):
            return x.Sa / x.Na + rho * np.sqrt(np.log(x.t + 1)*2 / x.Na)
        return self.Index_Policy(T, index_func)


    def weighted_bootstrap(self, T, gamma=1.0, explo_func=default_exp):
        """
        Implementation of the Vanilla Bootstrap bandit algorithm
        :param T: Time Horizon
        :param gamma: float, discount factor for the weights
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
        """
        assert gamma >= 0.0 and gamma <= 1.0, 'gamma should be between 0 and 1.'
        tr = Tracker2(
            self.means,
            T,
            store_rewards_arm=True,
            store_visit_times=True
            )
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                bts_mean = np.zeros(self.nb_arms)
                need_more_explo = (tr.Na < explo_func(t))
                if any(need_more_explo):
                    arm = np.random.choice(np.nonzero(need_more_explo)[0])
                else:
                    for k in range(self.nb_arms):
                        log_weights = (t - np.array(tr.visit_times[k])) * np.log(gamma)
                        weights = softmax(log_weights)
                        Na_weighted = int(np.ceil(np.sum(np.exp(log_weights))))
                        bts_mean[k] = np.mean(
                            np.random.choice(
                                tr.rewards_arm[k],
                                # size=int(tr.Na[k]),
                                size=Na_weighted,
                                replace=True,
                                p=weights,
                                ),
                            )
                    arm = rd_argmax(bts_mean)
            reward = self.MAB[arm].sample(t)[0]
            tr.update(t, arm, reward)
        return tr


    def weighted_SDA(self, T, gamma=1.0, explo_func=default_exp):
        """
        Implementation of weighted with replacement-SDA
        :param T: Time Horizon
        :param gamma: float, discount factor for the weights
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
        """
        assert gamma >= 0.0 and gamma <= 1.0, 'gamma should be between 0 and 1.'
        tr = Tracker2(
            self.means,
            T,
            store_rewards_arm=True,
            store_visit_times=True
            )
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            arm = t
            tr.update(t, arm, self.MAB[arm].sample(t)[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader_weighted(tr.Na, tr.Sa, tr.visit_times, t, gamma, l_prev)
            log_weights_leader = (t - np.array(tr.visit_times[l])) * np.log(gamma)
            weights_leader = softmax(log_weights_leader)
            Na_weighted_leader = int(np.ceil(np.sum(np.exp(log_weights_leader))))
            t_prev, forced_explo = t, explo_func(r)
            indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                log_weights = (t - np.array(tr.visit_times[j])) * np.log(gamma)
                weights = softmax(log_weights)
                Na_weighted = int(np.ceil(np.sum(np.exp(log_weights))))
                # if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                if indic[j] == 0 and j != l and Na_weighted < Na_weighted_leader:
                    lead_mean = np.mean(
                        np.random.choice(
                            tr.rewards_arm[l],
                            # size=int(tr.Na[j]),
                            size=Na_weighted,
                            replace=True,
                            p=weights_leader,
                            ),
                        )
                    challenger_mean = np.mean(
                        np.random.choice(
                            tr.rewards_arm[j],
                            # size=int(tr.Na[j]),
                            size=Na_weighted,
                            replace=True,
                            p=weights,
                            ),
                        )
                    # challenger_mean = tr.Sa[j] / tr.Na[j]
                    if challenger_mean >= lead_mean and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                tr.update(t, l, self.MAB[l].sample(t)[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        tr.update(t, i, self.MAB[i].sample(t)[0])
                        t += 1
            r += 1
        return tr

    def LB_SDA(self, T, gamma=1.0, explo_func=default_exp):
        """
        Implementation of the LB-SDA algorithm
        :param T: Time Horizon
        :param explo_func: Forced exploration function
        :return: Tracker object with the results of the run
        """
        assert gamma >= 0.0 and gamma <= 1.0, 'gamma should be between 0 and 1.'
        tr = Tracker2(
            self.means,
            T,
            store_rewards_arm=True,
            store_visit_times=True
            )
        r, t, l = 1, 0, -1
        while t < self.nb_arms:
            arm = t
            tr.update(t, arm, self.MAB[arm].sample(t)[0])
            t += 1
        while t < T:
            l_prev = l
            l = get_leader_weighted(tr.Na, tr.Sa, tr.visit_times, t, gamma, l_prev)
            log_weights_leader = (t - np.array(tr.visit_times[l])) * np.log(gamma)
            Na_weighted_leader = int(np.ceil(np.sum(np.exp(log_weights_leader))))
            t_prev, forced_explo = t, explo_func(r)
            # indic = (tr.Na < tr.Na[l]) * (tr.Na < forced_explo) * 1.
            indic = (tr.Na < forced_explo) * 1.
            for j in range(self.nb_arms):
                log_weights = (t - np.array(tr.visit_times[j])) * np.log(gamma)
                Na_weighted = int(np.ceil(np.sum(np.exp(log_weights))))
                indic[j] *= (Na_weighted < Na_weighted_leader)
                if indic[j] == 0 and j != l and tr.Na[j] < tr.Na[l]:
                    # lead_mean = np.mean(tr.rewards_arm[l][-int(tr.Na[j]):])
                    lead_mean = np.mean(tr.rewards_arm[l][-Na_weighted:])
                    if tr.Sa[j]/tr.Na[j] >= lead_mean and t < T:
                        indic[j] = 1
            if indic.sum() == 0:
                tr.update(t, l, self.MAB[l].sample(t)[0])
                t += 1
            else:
                to_draw = np.where(indic == 1)[0]
                np.random.shuffle(to_draw)
                for i in to_draw:
                    if t < T:
                        tr.update(t, i, self.MAB[i].sample(t)[0])
                        t += 1
            r += 1
        return tr
