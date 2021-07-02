# AUTHOR SPENCER LOGGIA
# INIT DATE 06/30/2021
from typing import Tuple

import numpy as np
from scipy import stats


class GenModel:

    def __init__(self, rewards: int = 4, modalities: int = 2, space_dims=2):
        self.mu = np.zeros((modalities, rewards, space_dims, 1))
        self.sigma = np.zeros((modalities, rewards, space_dims, space_dims))
        self.num_examples = np.zeros((modalities, rewards))
        self.modalities = modalities
        self.reward_freq = np.zeros(rewards)
        self.rewards = rewards
        self.trial_history = []

    def fit_trial(self, x: np.ndarray, y: int):
        """

        :param x: dim 0: space, dim 1: axis
        :param y: reward type / gmm component
        :return:
        """
        self.reward_freq[y] += 1
        self.trial_history.append((x, y))
        self.num_examples[:, y] += 1
        num_trials = self.num_examples[:, y]
        self.mu[:, y, :] = self.mu[:, y, :] + (x - self.mu[:, y, :]) / num_trials
        old_var = self.sigma[:, y, :, :]
        self.sigma[:, y, :, :] = old_var + (((x - self.mu) @ (x - self.mu).T) - old_var) / num_trials


    def predict_trial(self, x: np.ndarray, modality: int):
        """

        :param x: batch x shape dim
        :return: posterior probabilities of each reward value based on this modality
        """
        p_x_r = np.zeros(self.rewards)

        for r in range(self.rewards):
            p_x_r[r] = stats.multivariate_normal.cdf(x, self.mu[modality, r, :], cov=self.sigma[modality, r, :, :])
        p_x = np.sum(p_x_r)
        p_r = self.reward_freq / np.sum(self.reward_freq)
        p_r_x = (p_x_r * p_r) / p_x
        return p_r_x


    def integrator(self, reward_prob: np.ndarray) -> Tuple[float, float]:
        """
        optimally integrate the reward estimates from individual cues to estimate reward.
        :param reward_probs: From each modality the probabilities for each reward (m x r)
        :return tuple of combined reward probabilities:
        """
        # TODO: Implement this as rewards weighted by probs (can we just softmax the sum of the distributions? )
        pass


class SpaceSampler:
    #
    def __init__(self, seed=None):
        pass

    def generate_samples(self):
        """
        draw samples from the space and return expected reward and variance.
        :return:
        """
        yield

class StimuliSpace:
    """
    A space contains n dim vectors of object ids, coordinates, and gaussian mean coordinates, gaussian variance, and
    associated reward value
    """
    def __init__(self, ):


def simulate(trials=100):
    color_space = SpaceSampler() # needs args
    shape_space = SpaceSampler() # needs args
    cs_gen = color_space.generate_samples()
    ss_gne = shape_space.generate_samples()
    for i in range(trials):






