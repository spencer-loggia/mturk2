import sys

from mturk2_code.sim import ColorShapeData
import dropbox
import numpy as np
import pandas as pd


class Agent:

    def __init__(self, shape_axis_size: int, color_axis_size: int, decision_policy: str):
        self.shape_axis_size = shape_axis_size
        self.color_axis_size = color_axis_size
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

    def get_reward_marginals(self, axis: str, reward_matrix):
        dim = self._get_dim(axis)
        num_rewards = len(np.unique(reward_matrix))
        num_items = reward_matrix.shape[dim]
        marginals = np.empty((num_items, num_rewards))
        for i in range(num_rewards):
            marginals[:, i] = np.count_nonzero(reward_matrix == i, axis=dim) / num_items
        return marginals

    def get_frequency_marginals(self, axis: str, freq_matrix):
        dim = self._get_dim(axis)
        num_items = freq_matrix.shape[dim]
        marginals = np.sum(freq_matrix, axis=dim) / num_items
        return marginals

    def _init_bli(self, reward_matrix):
        reward_shape_marginal = self.get_reward_marginals('shape', reward_matrix)
        reward_color_marginals = self.get_reward_marginals('color', reward_matrix)
        num_shapes = reward_shape_marginal.shape[0]
        num_colors = reward_color_marginals.shape[0]
        numer = reward_shape_marginal[None, :, :] * reward_color_marginals[:, None, :]
        rewards, reward_counts = np.unique(reward_matrix, return_counts=True)
        reward_dist = reward_counts / (num_shapes * num_colors)
        p_r_sc = numer / reward_dist[None, None, :]
        return p_r_sc

    def _init_conv_net(self):
        try:
            from torchvision.models import alexnet
        except ModuleNotFoundError:
            print("Torchvision Module Not Found. Please install Torch and Torchvision.", sys.stderr)
            exit(-3)

    def fit(self, x):
        """
        Either sets parameters from training data or initialize proper distributions, depending on mode
        """
        if self.decision_policy == 'bli':
            try:
                reward_matrix = x.rewards.reshape(self.shape_axis_size, self.color_axis_size)
                self.freq_matrix = x.freq.reshape(self.shape_axis_size, self.color_axis_size)
            except AttributeError:
                print('Provided dataset \'x\' must be of type ColorSpaceData.')
                exit(-1)
            self.p_r_sc = self._init_bli(reward_matrix)

        elif self.decision_policy == 'optimal':
            self.reward_matrix = x.rewards.reshape(self.shape_axis_size, self.color_axis_size)
        else:
            exit(-2)

    def predict(self, shape_index, color_index):
        if self.decision_policy == 'bli':
            if not hasattr(self, 'p_r_sc'):
                raise AttributeError('Must have run \'fit\' with decision policy \'bli\'')
            else:
                reward_dist = self.p_r_sc[shape_index, color_index]
                return np.argmax(reward_dist)
        if self.decision_policy == 'optimal':
            if not hasattr(self, 'reward_matrix'):
                raise AttributeError('Must have run \'fit\' with decision policy \'optimal\'')
            else:
                return self.reward_matrix[shape_index, color_index]


if __name__ == '__main__':
    data = ColorShapeData('../../data/images/imp0.png',
                          '../../data/reward_space.csv',
                          '../../data/freq_space.csv',
                          num_samples=36 * 36)

    ag = Agent(36, 36, decision_policy='optimal')
    ag.fit(data)
    print('done')
