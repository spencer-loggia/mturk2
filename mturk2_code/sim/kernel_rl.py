import torch
import pandas as pd
import numpy as np
from neurotools import util

class KernelRL:
    def __init__(self, device='cpu', precision="single", resolution=72, temporal_window=20, expected_plateu=100000, kernel_mode="logistic"):
        """
        Fits monkey behavior by updating with a parameterized gaussian kernel.
        """
        self.s_real = 36
        self.c_real = 36
        self.resolution = round(resolution)
        self.dynamic_kernel = kernel_mode == "logistic"

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
        self.init_kernel_cov = torch.nn.Parameter(torch.tensor([.5, .01, .5], device=device, dtype=self.ndtype))
        if self.dynamic_kernel:
            # compute initial tau to reach 90% final at 1/2 number of trials 
            tau = -2 * torch.log(torch.tensor([.1 / 1.9], dtype=self.ndtype, device=device)) / expected_plateu
            self.kernel_log_tau = torch.nn.Parameter(torch.log(tau))
            self.final_kernel_cov = torch.nn.Parameter(self.init_kernel_cov.clone())

        self.log_lr = torch.nn.Parameter(torch.tensor([-2], device=device, dtype=self.ndtype))

        self.inverse_temperature = torch.nn.Parameter(torch.ones(1, device=device, dtype=self.ndtype))
        
        self.state_spaces = []
        self.loss_hist = []

        self.optimizer = None
        
    def to(self, device):
        self.init_prob = torch.nn.Parameter(self.init_prob.to(device))
        self.init_kernel_cov = torch.nn.Parameter(self.init_kernel_cov.to(device))
        if self.dynamic_kernel:
            self.kernel_log_tau = torch.nn.Parameter(self.kernel_log_tau.to(device))
            self.final_kernel_cov = torch.nn.Parameter(self.final_kernel_cov.to(device))
        self.log_lr = torch.nn.Parameter(self.log_lr.to(device))
        self.inverse_temperature = torch.nn.Parameter(self.inverse_temperature.to(device))
    
    def _get_choices_rewards(self, trial_data):
        """
        <n, 4, 2> choices on range (0, s_real | c_real)
        """
        raw_choices = trial_data[["c1", "s1", "c2", "s2", "c3", "s3", "c4", "s4"]].to_numpy().reshape((-1, 4, 2))
        raw_choices = torch.from_numpy(raw_choices).to(self.ndtype).to(self.device)
        choices = raw_choices / torch.tensor([self.c_real, self.s_real], dtype=self.ndtype, device=self.device)[None, None, :]
        choice_idx = torch.from_numpy(trial_data.choice_idx.to_numpy()).long().to(self.device)
        reward = torch.from_numpy(trial_data.reward.to_numpy()).to(self.ndtype).to(self.device)
        return torch.round(self.resolution * choices).int(), choice_idx, reward

    def _nonsingular_cov_from_triu(self, vec):
        cov_in = util.triu_to_square(vec, 2, includes_diag=True)
        return cov_in.clone() @ cov_in.clone()

    def fit(self, trial_data:pd.DataFrame, epochs=1000, lr=.01):
        """
        trial_data: DataFrame: cols: s1-s4, c1-c4, reward, choice_idx
        """
        ce = torch.nn.CrossEntropyLoss()
        state_spaces = None
        options, choice_idxs, rewards = self._get_choices_rewards(trial_data)
        if not self.dynamic_kernel:
            self.optimizer = torch.optim.Adam(params=[self.init_prob, self.init_kernel_cov,
                                                      self.inverse_temperature, self.log_lr],
                                              lr=lr)
        else:
            self.optimizer = torch.optim.Adam(params=[self.init_kernel_cov, self.kernel_log_tau,
                                                      self.final_kernel_cov, self.inverse_temperature, self.log_lr],
                                              lr=lr)
        for epoch in range(epochs):
            loss = torch.zeros((1,), dtype=self.ndtype, device=self.device)
            self.state_spaces = []
            print("Epoch", epoch)
            self.optimizer.zero_grad()
            state_space = self.init_prob.clone()
            # iterate through all trials

            for ind in range(len(trial_data)):
                # build true nonsingular cov from cov input
                kernel_cov = self._nonsingular_cov_from_triu(self.init_kernel_cov)
                if self.dynamic_kernel:
                    f_kernel_cov = self._nonsingular_cov_from_triu(self.final_kernel_cov)
                    # select the input covariance matrix at this time point
                    kernel_cov = util.logistic_func(torch.tensor([ind]), kernel_cov.flatten(),
                                                 f_kernel_cov.flatten(), self.kernel_log_tau).reshape((2, 2))
                # construct gaussian kernel (uses spencer's tools library)
                kernel = util.gaussian_kernel(kernel_size=(self.resolution, self.resolution),
                                              cov=kernel_cov * self.resolution,
                                              integral_resolution=1,
                                              renormalize=True,)
                # to (0, 1)
                reward = rewards[ind]
                # choice_idx = int(trial.choice_idx)
                # get the expected reward of each option from most recent state space
                option_reward_exp = state_space[options[ind, :, 0], options[ind, :, 1]]
                # compute softmax
                choice_probs = torch.softmax(option_reward_exp * self.inverse_temperature, dim=0)
                # compare to monkey choice
                choice_idx = choice_idxs[ind]
                loss = loss - choice_probs[choice_idx]
                # add state space to list
                self.state_spaces.append(state_space.clone())
                # # update reward space
                offset_mu = (options[ind, choice_idx] - self.resolution // 2).detach().cpu().tolist() # how much to translate kernel
                state_space = self.state_spaces[-1].clone()
                # # center kernel on choice
                centered_kernel = torch.roll(kernel.clone(), offset_mu, dims=(0, 1))
                state_space = state_space + torch.exp(self.log_lr) * centered_kernel * (reward.detach() - state_space)  # kernel need to wieght both observed reward and current space
            # propagate gradient
            loss.backward(retain_graph=False)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_hist.append(loss.detach().cpu().item())
            # check early stopping:
            if epoch > 50:
                states = self.optimizer.state_dict()["state"]
                moments = []
                for p in states.keys():
                    moments.append(states[p]["exp_avg"].flatten())
                moment = torch.abs(torch.concat(moments, dim=0)).mean()
                print(moment)
                if moment < 1e-1:
                    print("Stopping early...")
                    break

        return state_spaces

    def get_results(self):
        res_dict = {}
        res_dict["epoch_loss_history"] = np.array(self.loss_hist)
        res_dict["init_kernel_cov"] = self._nonsingular_cov_from_triu(self.init_kernel_cov).detach().cpu().numpy().reshape((2, 2))
        if self.dynamic_kernel:
            res_dict["kernel_log_tau"] = self.kernel_log_tau.detach().cpu().numpy()
            res_dict["final_kernel_cov"] = self._nonsingular_cov_from_triu(self.final_kernel_cov).detach().cpu().numpy().reshape((2, 2))
        else:
            res_dict["kernel_log_tau"] = None
            res_dict["final_kernel_cov"] = None
        res_dict["log_learn_rate"] = self.log_lr.detach().cpu().numpy()
        res_dict["inverse_temperature"] = self.inverse_temperature.detach().cpu().numpy()
        res_dict["initial_value_space"] = self.init_prob.detach().cpu().numpy()
        res_dict["state_spaces"] = [space.detach().cpu().numpy() for space in self.state_spaces]
        return res_dict
    
    






