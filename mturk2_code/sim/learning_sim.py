import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from dataset import ColorShapeData
from policy import ConvNet
from matplotlib import pyplot as plt


class Agent:

    def __init__(self, policy: nn.Module,
                 lr_init: float = .01, lr_decay_iter: int = 2000, lr_decay_scale: float = .2,):
        self.brain = policy
        self.optimizer = SGD(params=self.brain.parameters(), lr=lr_init)
        self.lr_scheduler = StepLR(self.optimizer, step_size=lr_decay_iter, gamma=lr_decay_scale)

    def threshold(self, val: torch.Tensor):
        val = np.array(val.cpu().detach())
        result = np.zeros(val.shape)
        result[np.nonzero(val < -1.5)] = 0
        result[np.nonzero(np.logical_and(-1.5 <= val, val < -.5))] = 1
        result[np.nonzero(np.logical_and(-.5 <= val, val < .5))] = 2
        result[np.nonzero(.5 <= val)] = 3
        return torch.from_numpy(result)

    def learn(self, data: ColorShapeData):
        reward_hist = []
        sliding_hist = []
        self.optimizer.zero_grad()
        count = 0
        loss = torch.nn.MSELoss()
        while data.has_next():
            stimuli, gt_rewards = data.next_trial(num_to_pick=16)
            stimuli = torch.from_numpy((np.stack(stimuli))).float()
            # epsilon greedy action selection
            self.optimizer.zero_grad()
            pred_rewards = self.brain(stimuli).reshape(-1)
            count += 16
            gt_rewards = torch.tensor(gt_rewards, dtype=torch.float)
            l = loss(pred_rewards, gt_rewards)
            sliding_hist.append(l.item())
            if (count) % 160 == 0:
                print('trial', count, ': loss over cycle :', l, 'gt:', gt_rewards[0:5], 'pred:', self.threshold(pred_rewards[0:5] - 2))
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        return sliding_hist


if __name__ == '__main__':
    import pickle

    data = ColorShapeData('../../data/images/imp0.png',
                          '../../data/reward_space.csv',
                          '../../data/freq_space.csv',
                          num_samples=40000)
    model = ConvNet()
    agent = Agent(policy=model)
    hist = agent.learn(data)
    plt.plot(hist)
    plt.show()
    with open('../saved_data/net_saved.pkl', 'wb') as f:
        pickle.dump(agent.brain, f)
