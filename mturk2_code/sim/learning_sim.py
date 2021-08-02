import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from dataset import ColorShapeData
from policy import QNet
from matplotlib import pyplot as plt


class Agent:

    def __init__(self, policy: nn.Module,
                 lr_init: float = .0001, lr_decay_iter: int = 2000, lr_decay_scale: float = .5,
                 eplison: float = .95, epsilon_decay_pattern: str = 'linear', epsilon_final: float = 0.,
                 gamma_temporal_decay: float = .9):
        self.brain = policy
        self.optimizer = SGD(params=self.brain.parameters(), lr=lr_init)
        self.lr_scheduler = StepLR(self.optimizer, step_size=lr_decay_iter, gamma=lr_decay_scale)
        self.gamma = torch.tensor([gamma_temporal_decay], requires_grad=False, dtype=torch.float)
        self.init_epsilon = eplison
        self.epsilon_decay_pattern = epsilon_decay_pattern
        self.final_epsilon = epsilon_final

    @staticmethod
    def functional_epsilon(init, pattern, final, steps, x):
        m = (final - init) / steps
        if pattern == 'linear':
            eps = init + m * x
        elif pattern == 'rational':
            a = final * steps * init / (1 - final)
            eps = 2 * (a / (x + (a / init)) + (init + m * x)) / 3
        else:
            raise ValueError
        return eps

    def learn(self, data: ColorShapeData):
        reward_hist = []
        sliding_hist = []
        self.optimizer.zero_grad()
        count = 0
        loss = torch.nn.CrossEntropyLoss()
        while data.has_next():
            stimuli, gt_rewards = data.next_trial(num_to_pick=40)
            stimuli = torch.from_numpy((np.stack(stimuli))).float()
            # epsilon greedy action selection
            self.optimizer.zero_grad()
            pred_rewards = self.brain(stimuli).reshape(-1, 4)
            count += 40
            gt_rewards = torch.tensor(gt_rewards, dtype=torch.int64)
            l = loss(pred_rewards,gt_rewards)
            sliding_hist.append(l.item())
            reward_hist.append(torch.count_nonzero(torch.argmax(pred_rewards, dim=1) == gt_rewards) / len(gt_rewards))
            if (count) % 120 == 0:
                print('trial', count, ': loss over cycle :', l, 'acc over cycle: ', reward_hist[-1])
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        return sliding_hist


if __name__ == '__main__':
    import pickle

    data = ColorShapeData('../../data/images/imp0.png',
                          '../../data/reward_space.csv',
                          '../../data/freq_space.csv',
                          num_samples=20000)
    model = QNet()
    agent = Agent(policy=model, epsilon_decay_pattern='rational', epsilon_final=.01, gamma_temporal_decay=.5)
    hist = agent.learn(data)
    plt.plot(hist)
    plt.show()
    with open('../saved_data/net_saved.pkl', 'wb') as f:
        pickle.dump(agent.brain, f)
