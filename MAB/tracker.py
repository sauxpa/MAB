import numpy as np
from bisect import bisect, insort


class Tracker2:
    """
    This object is used in bandit models to store useful quantities to run
    the algorithm and report the experiment.
    """
    def __init__(self,
                 means,
                 T,
                 alpha=None,
                 risk_measure='mean',
                 risk_measures=[],
                 store_rewards_arm=False,
                 store_visit_times=False,
                 store_sorted_rewards_arm=False,
                 ):
        self.means = means

        self.risk_measure = risk_measure

        # Risk aversion parameter
        self.alpha = alpha

        self.risk_measures = risk_measures

        self.nb_arms = means.shape[0]
        self.T = T
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.empty(self.T, dtype=int)
        self.t = 0
        self.store_rewards_arm = store_rewards_arm
        self.store_visit_times = store_visit_times
        self.store_sorted_rewards_arm = store_sorted_rewards_arm

        if store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]
        if store_visit_times:
            self.visit_times = [[] for _ in range(self.nb_arms)]
        if store_sorted_rewards_arm:
            self.sorted_rewards_arm = [[] for _ in range(self.nb_arms)]
            self.Na_quantile = np.zeros(self.nb_arms)
            self.idx_quantile = np.zeros(self.nb_arms)

    def reset(self):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward of arm a
                 - Na: np.array, number of times arm a has been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chose at each step
        """
        self.Sa = np.zeros(self.nb_arms)
        self.Na = np.zeros(self.nb_arms)
        self.reward = np.zeros(self.T)
        self.arm_sequence = np.zeros(self.T, dtype=int)
        self.rewards_arm = [[]]*self.nb_arms
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_visit_times:
            self.visit_times = [[] for _ in range(self.nb_arms)]
        if self.store_sorted_rewards_arm:
            self.sorted_rewards_arm = [[] for _ in range(self.nb_arms)]
            self.Na_quantile = np.zeros(self.nb_arms)
            self.idx_quantile = np.zeros(self.nb_arms)

    def update(self, t, arm, reward):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        self.Na[arm] += 1
        self.arm_sequence[t] = arm
        self.reward[t] = reward
        self.Sa[arm] += reward
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if self.store_visit_times:
            self.visit_times[arm].append(self.t)
        if self.store_sorted_rewards_arm:
            # Insert new reward to the histogram for corresponding action.
            idx_reward = bisect(self.sorted_rewards_arm[arm], reward)
            self.idx_quantile[arm] = np.ceil(
                self.alpha * self.Na[arm]
                ).astype(int)
            insort(self.sorted_rewards_arm[arm], reward)

            # Update number of visit to the alpha quantile of arm
            if idx_reward <= self.idx_quantile[arm]:
                self.Na_quantile[arm] += 1

    def regret(self, regret_mode='mean'):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        if regret_mode == 'mean':
            return self.means.max() * np.arange(1, self.T + 1) - np.cumsum(np.array(self.means)[self.arm_sequence])
        else:
            rhos = self.risk_measures[self.risk_measure]
            return rhos.max() * np.arange(1, self.T + 1) - np.cumsum(np.array(rhos)[self.arm_sequence])

    def regret_dynamic(self, regret_mode='mean'):
        """
        Compute the dynamic regret (i.e when mean can change throught time) of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        R = np.zeros(self.T)
        if regret_mode == 'mean':
            for t in range(self.T):
                R[t] = np.max([self.means[k](t) for k in range(self.nb_arms)]) - self.means[self.arm_sequence[t]](t)
            return np.cumsum(R)
        else:
            raise ValueError('Only available regret mode in dynamic regret is mean')
