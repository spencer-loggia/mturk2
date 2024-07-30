import torch
import pandas as pd
import numpy as np
from neurotools import util

class KernelRL:
    def __init__(self, device='cpu', precision="single", resolution=72, temporal_window=20, expected_plateu=100000, kernel_mode="exp", forget_mode="stable"):
        """
        Fits monkey behavior by updating with a parameterized gaussian kernel.
        All time measured in hours
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

        self.init_prob = torch.nn.Parameter(torch.zeros((resolution, resolution), device=device, dtype=self.ndtype))
        self.init_kernel_cov = torch.nn.Parameter(torch.tensor([.5, 0.0, .5], device=device, dtype=self.ndtype))
        # compute initial tau to reach 63% final at 1/3 total trials
        tau = -torch.log(torch.tensor([.63 * expected_plateu], dtype=self.ndtype, device=device))
        self.kernel_log_tau = torch.nn.Parameter(torch.log(tau))
        self.final_kernel_cov = torch.nn.Parameter(self.init_kernel_cov.clone())

        self.forget_mode = forget_mode

        # time constant for returning from current values state to initial (hours)
        # start with being 63% forgotten after 4 days.
        self.log_init_forget_tau = torch.nn.Parameter(-torch.log(torch.tensor([24.*4.*.63],
                                                                              device=device, dtype=self.ndtype)))
        self.log_final_forget_tau = torch.nn.Parameter(-torch.log(torch.tensor([24.*60.*.63],
                                                                               device=device, dtype=self.ndtype)))
        # the acceleration of forgetting, in trials.
        self.log_forget_tau_tau = torch.nn.Parameter(-torch.log(torch.tensor([.63 * expected_plateu],
                                                                               device=device, dtype=self.ndtype)))

        self.log_lr = torch.nn.Parameter(torch.tensor([-2], device=device, dtype=self.ndtype))

        self.inverse_temperature = torch.nn.Parameter(torch.ones(1, device=device, dtype=self.ndtype))
        
        self.state_spaces = []
        self.loss_hist = []

        self.optimizer = None
        
    def to(self, device):
        self.init_prob = torch.nn.Parameter(self.init_prob.to(device))
        self.init_kernel_cov = torch.nn.Parameter(self.init_kernel_cov.to(device))
        self.kernel_log_tau = torch.nn.Parameter(self.kernel_log_tau.to(device))
        self.final_kernel_cov = torch.nn.Parameter(self.final_kernel_cov.to(device))
        self.log_init_forget_tau = torch.nn.Parameter(self.log_init_forget_tau.to(device))
        self.log_final_forget_tau = torch.nn.Parameter(self.log_final_forget_tau.to(device))
        self.log_forget_tau_tau = torch.nn.Parameter(self.log_forget_tau_tau.to(device))
        self.log_lr = torch.nn.Parameter(self.log_lr.to(device))
        self.inverse_temperature = torch.nn.Parameter(self.inverse_temperature.to(device))
    
    def _get_choices_rewards(self, trial_data):
        """
        <n, 4, 2> choices on range (0, s_real | c_real)
        """
        raw_choices = trial_data[["c1", "s1", "c2", "s2", "c3", "s3", "c4", "s4"]].to_numpy().reshape((-1, 4, 2))
        time_stamps = trial_data[["time_delta_seconds"]].to_numpy()
        time_stamps = time_stamps - np.concatenate([time_stamps[0], time_stamps[:-1]]) # every time stamp minus the one before
        time_stamps = time_stamps / (60 * 60) # convert from seconds to hours.
        raw_choices = torch.from_numpy(raw_choices).to(self.ndtype).to(self.device)
        choices = raw_choices / torch.tensor([self.c_real, self.s_real], dtype=self.ndtype, device=self.device)[None, None, :]
        choice_idx = torch.from_numpy(trial_data.choice_idx.to_numpy()).long().to(self.device)
        reward = torch.from_numpy(trial_data.reward.to_numpy()).to(self.ndtype).to(self.device)
        return torch.round(self.resolution * choices).int(), choice_idx, reward, time_stamps

    def _nonsingular_cov_from_triu(self, vec):
        cov_in = util.triu_to_square(vec, 2, includes_diag=True)
        return cov_in.clone() @ cov_in.clone()

    def fit(self, trial_data:pd.DataFrame, epochs=1000, lr=.01):
        """
        trial_data: DataFrame: cols: s1-s4, c1-c4, reward, choice_idx
        """
        ce = torch.nn.CrossEntropyLoss()
        state_spaces = None
        options, choice_idxs, rewards, time_delta = self._get_choices_rewards(trial_data)

        self.optimizer = torch.optim.Adam(params=[self.init_kernel_cov, self.kernel_log_tau,
                                                  self.final_kernel_cov, self.inverse_temperature, self.log_lr,
                                                  self.log_init_forget_tau, self.log_final_forget_tau, self.log_forget_tau_tau],
                                          lr=lr)
        for epoch in range(epochs):
            loss = torch.zeros((1,), dtype=self.ndtype, device=self.device)
            self.state_spaces = []
            print("Epoch", epoch)
            self.optimizer.zero_grad()
            state_space = self.init_prob.clone()
            # build true nonsingular cov from cov input
            kernel_cov = self._nonsingular_cov_from_triu(self.init_kernel_cov)
            f_kernel_cov = self._nonsingular_cov_from_triu(self.final_kernel_cov)
            # select the input covariance matrix at this time point
            kernel_covs = util.exponential_func(torch.arange(len(trial_data)), kernel_cov.flatten(),
                                               f_kernel_cov.flatten(), self.kernel_log_tau).reshape((-1, 2, 2))
            # get the forget time constants for all trials
            log_forget_taus = torch.log(util.exponential_func(torch.arange(len(trial_data)),
                                                              torch.exp(self.log_init_forget_tau),
                                                              torch.exp(self.log_final_forget_tau),
                                                              self.log_forget_tau_tau).flatten())

            # iterate through all trials
            for ind in range(len(trial_data)):
                kernel_cov = kernel_covs[ind]
                time = time_delta[ind]
                log_forget_tau = log_forget_taus[ind]
                # compute how much we forgot since the last trial
                cur_state = util.exponential_func(torch.tensor([time]), state_space.flatten(),
                                                  self.init_prob.clone().flatten(),
                                                  log_forget_tau).reshape(state_space.shape)
                state_space = cur_state

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
        res_dict["kernel_log_tau"] = self.kernel_log_tau.detach().cpu().numpy()
        res_dict["final_kernel_cov"] = self._nonsingular_cov_from_triu(self.final_kernel_cov).detach().cpu().numpy().reshape((2, 2))
        res_dict["init_forget_log_tau"] = self.log_init_forget_tau.detach().cpu().numpy()
        res_dict["final_forget_log_tau"] = self.log_final_forget_tau.detach().cpu().numpy()
        res_dict["forget_log_tau_tau"] = self.log_forget_tau_tau.detach().cpu().numpy()
        res_dict["log_learn_rate"] = self.log_lr.detach().cpu().numpy()
        res_dict["inverse_temperature"] = self.inverse_temperature.detach().cpu().numpy()
        res_dict["initial_value_space"] = self.init_prob.detach().cpu().numpy()
        res_dict["state_spaces"] = [space.detach().cpu().numpy() for space in self.state_spaces]
        return res_dict
    
    







