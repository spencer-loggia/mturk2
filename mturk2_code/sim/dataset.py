from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from pandas import read_csv
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # windows truly is awful

from matplotlib import pyplot as plt, cm


class ColorShapeData(Dataset):

    def __init__(self, shape_data_path: str,
                 reward_assignment_path: str,
                 prob_stim_path: str,
                 color_spec_path: Union[str, None] = None,
                 num_samples: int = 1000):
        super().__init__()
        shapes = Image.open(shape_data_path)
        shapes = np.array(shapes).reshape((400, 36, 400, 3))
        self.shapes = np.transpose(shapes, (1, 0, 2, 3))
        self.rewards = read_csv(reward_assignment_path).iloc[:, 0].values.reshape(-1)
        exp_prob = np.exp(read_csv(prob_stim_path).iloc[:, 0].values.reshape(-1))
        self.freq = exp_prob / np.sum(exp_prob)
        self.num_samples = num_samples
        if color_spec_path is None:
            cspace = cm.get_cmap('twilight')
            idx = np.random.choice(np.arange(cspace.N), 36)
            idx.sort()
            self.colors = np.array(cspace.colors)[idx] * 255
        else:
            self.colors = np.array(read_csv(color_spec_path).values, dtype=float) * 255
        self.sample_idxs = np.random.choice(np.arange(len(self)), size=self.num_samples, p=self.freq)
        self.head = 0
        print('loaded data')

    def __len__(self):
        return len(self.shapes) * len(self.colors)

    def __getitem__(self, item):
        color_idx = np.floor(item / len(self.shapes))
        shape_idx = item % len(self.shapes)
        shape = self.shapes[shape_idx]
        collapsed = np.sum(shape, axis=2)
        color_area = np.nonzero(collapsed == 49 + 88 + 163)  # color the white area
        shape[color_area[0], color_area[1], :] = self.colors[int(color_idx)]
        shape = np.transpose(shape, (2, 0, 1))  # change to torch indexing (channel, h, w)
        return shape, self.rewards[item]

    def has_next(self):
        return self.head < self.num_samples - 3

    def next_trial(self, num_to_pick=4, advance=True):
        load_idx = self.sample_idxs[self.head: min(self.head+num_to_pick, self.num_samples-1)]
        if advance:
            self.head += num_to_pick
        stimuli = []
        rewards = []
        for idx in load_idx:
            stimulus, reward = self[idx]
            stimuli.append(stimulus)
            rewards.append(reward)
        stacked = np.stack(stimuli).flatten()
        mean = np.mean(stacked)
        std = np.std(stacked)
        stimuli = [(stimulus - mean) / std for stimulus in stimuli]
        return stimuli, rewards


if __name__ == '__main__':
    test = ColorShapeData('../../data/images/imp0.png', '../../data/reward_space.csv', '../../data/freq_space.csv')
    item = test[692]
    downsample = torch.nn.AvgPool2d(8)
    item = downsample(torch.from_numpy(item[0]).float()[None, :, :, :])
    item = np.array(item) / 255
    plt.imshow(item[0].transpose(1, 2, 0))
    plt.show()
    print('Expected random reward', np.sum(test.freq * test.rewards))
