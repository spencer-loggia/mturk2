from mturk2_code.sim import ColorShapeData
import dropbox
import numpy as np
import pandas as pd


class Agent:

    def __init__(self, dataset: ColorShapeData, shape_axis_size: int, color_axis_size: int, decision_policy: str):
        self.reward_matrix = dataset.rewards.reshape(shape_axis_size, color_axis_size)
        self.freq_matrix = dataset.freq.reshape(shape_axis_size, color_axis_size)
        self.decision_policy = decision_policy

    @staticmethod
    def _get_dim(axis: str):
        if axis == 'shape':
            dim = 0
        elif axis == 'color':
            dim = 1
        else:
            raise ValueError
        return dim

    def get_reward_marginals(self, axis: str):
        dim = self._get_dim(axis)
        num_rewards = len(np.unique(self.reward_matrix))
        num_items = self.reward_matrix.shape[dim]
        marginals = np.empty((num_items, num_rewards))
        for i in range(num_rewards):
            marginals[:, i] = np.count_nonzero(self.reward_matrix == i, axis=dim) / num_items
        return marginals

    def get_frequency_marginals(self, axis: str):
        dim = self._get_dim(axis)
        num_items = self.reward_matrix.shape[dim]
        marginals = np.sum(self.freq_matrix, axis=dim) / num_items
        return marginals

    def fit(self, x=None):
        """
        Either sets parameters from training data or initialize proper distributions, depending on mode
        """
        if self.decision_policy == 'linear_independent_integrator':
            reward_shape_marginal = self.get_reward_marginals(axis='shape')
            reward_color_marginals = self.get_reward_marginals(axis='color')
            num_shapes = reward_shape_marginal.shape[0]
            num_colors = reward_color_marginals.shape[0]
            numer = reward_shape_marginal[None, :, :] * reward_color_marginals[:, None, :]
            rewards, reward_counts = np.unique(self.reward_matrix, return_counts=True)
            reward_dist = reward_counts / (num_shapes * num_colors)
            p_r_sc = numer / reward_dist[None, None, :]
            self.p_r_sc = p_r_sc
        else:
            raise NotImplementedError

    def predict(self, shape_index, color_index):
        if self.decision_policy == 'linear_independent_integrator':
            if not hasattr(self, 'p_r_sc'):
                raise AttributeError('Must have run \'fit\' with decision policy \'linear_independent_integrator\'')
            else:
                reward_dist = self.p_r_sc[shape_index, color_index]
                return np.argmax(reward_dist)


if __name__ == '__main__':
    data = ColorShapeData('../../data/images/imp0.png',
                          '../../data/reward_space.csv',
                          '../../data/freq_space.csv',
                          num_samples=36 * 36)

    ag = Agent(data, 36, 36, decision_policy='linear_independent_integrator')
    ag.fit()
