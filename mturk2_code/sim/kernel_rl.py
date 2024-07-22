import torch
import pandas as pd
import numpy as np
from neurotools import util

class KernelRL:
    def __init__(self, device='cpu', precision="single", resolution=72, temporal_window=20):
        """
        Fits monkey behavior by updating with a parameterized gaussian kernel.
        """
        self.s_real = 36
        self.c_real = 36
        self.resolution = round(resolution)

        if precision == "single":
            self.ndtype = torch.float32
        elif precision == "double":
            self.ndtype = torch.float64
        else:
            raise ValueError

        self.device = device

        self.temporal_window = temporal_window

        self.init_prob = torch.nn.Parameter(torch.zeros((resolution,
                                                         resolution), device=device, dtype=self.ndtype))
        self.kernel_cov = torch.nn.Parameter(torch.eye(2, device=device, dtype=self.ndtype) * .5)

        self.log_lr = torch.nn.Parameter(torch.tensor([-2], device=device, dtype=self.ndtype))

        self.inverse_temperature = torch.nn.Parameter(torch.ones(1, device=device, dtype=self.ndtype))

        self.optimizer = torch.optim.Adam(params=[self.init_prob, self.kernel_cov, self.log_lr, self.inverse_temperature], lr=.1)

    def fit(self, trial_data:pd.DataFrame, epochs=1000):
        """
        trial_data: DataFrame: cols: s1-s4, c1-c4, reward, choice_idx
        """
        ce = torch.nn.CrossEntropyLoss()
        state_spaces = None
        for epoch in range(epochs):
            loss = torch.zeros((1,), dtype=self.ndtype, device=self.device)
            state_spaces = []
            print("Epoch", epoch)
            self.optimizer.zero_grad()
            state_space = self.init_prob.clone()
            # construct gaussian kernel (uses spencer's tools library)
            for ind, trial in trial_data.iterrows():
                print("trial", ind)
                kernel = util.gaussian_kernel(kernel_size=(self.resolution, self.resolution),
                                              cov=self.kernel_cov * self.resolution,
                                              integral_resolution=1,
                                              renormalize=False)
                raw_choices = torch.tensor([[trial.c1, trial.s1],
                                       [trial.c2, trial.s2],
                                       [trial.c3, trial.s3],
                                       [trial.c4, trial.s4]], dtype=self.ndtype, device=self.device)
                # to (0, 1)
                choices = raw_choices / torch.tensor([self.c_real, self.s_real]).unsqueeze(0)
                reward = torch.tensor([trial.reward], dtype=self.ndtype, device=self.device)
                choice_idx = int(trial.choice_idx)
                # get the expected reward of each option from most recent state space
                state_space_indexes = torch.round(choices * self.resolution).int()
                option_reward_exp = state_space[state_space_indexes[:, 0], state_space_indexes[:, 1]]
                # compute softmax
                choice_probs = torch.softmax(option_reward_exp * self.inverse_temperature, dim=0)
                # compare to monkey choice
                loss = loss + ce(choice_probs, torch.tensor(choice_idx))
                # add state space to list
                state_spaces.append(state_space.clone())
                # # update reward space
                mu = state_space_indexes[choice_idx].detach().tolist()
                state_space = state_spaces[-1].clone()
                # # center kernel on choice
                centered_kernel = torch.roll(kernel.clone(), mu, dims=(0, 1))
                state_space = state_space + torch.exp(self.log_lr) * (centered_kernel * reward.detach() - state_space)

            # propagate gradient
            loss.backward(retain_graph=False)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return state_spaces







