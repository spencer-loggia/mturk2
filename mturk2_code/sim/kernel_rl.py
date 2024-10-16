import torch
import pandas as pd
import numpy as np
from neurotools import util
from rl_base import BaseRL


class KernelRL(BaseRL):
    def __init__(self, device='cpu', precision="single", resolution=72, temporal_window=20, expected_plateu=100000,
                 kernel_mode="exp", forget_mode="stable"):
        """
        Fits monkey behavior by updating with a parameterized gaussian kernel.
        All time measured in hours
        """
        super().__init__(precision, device)
        self.resolution = round(resolution)
        self.dynamic_kernel = kernel_mode == "logistic"

        self.temporal_window = temporal_window

        self.init_prob = torch.nn.Parameter(torch.zeros((resolution, resolution), device=device, dtype=self.ndtype))
        self.init_kernel_cov = torch.nn.Parameter(torch.tensor([.5, 0.0, .5], device=device, dtype=self.ndtype))
        # compute initial tau to reach 63% final at 1/3 total trials
        tau = torch.tensor([.63 * expected_plateu], dtype=self.ndtype, device=device)
        self.kernel_log_tau = torch.nn.Parameter(-torch.log(tau))
        self.final_kernel_cov = torch.nn.Parameter(self.init_kernel_cov.clone())

        self.forget_mode = forget_mode

        # time constant for returning from current values state to initial (hours)
        # start with being 63% forgotten after 4 days.
        self.log_init_forget_tau = torch.nn.Parameter(-torch.log(torch.tensor([24. * 4. * .63],
                                                                              device=device, dtype=self.ndtype)))
        self.log_final_forget_tau = torch.nn.Parameter(-torch.log(torch.tensor([24. * 60. * .63],
                                                                               device=device, dtype=self.ndtype)))
        # the acceleration of forgetting, in trials.
        self.log_forget_tau_tau = torch.nn.Parameter(-torch.log(torch.tensor([.63 * expected_plateu],
                                                                             device=device, dtype=self.ndtype)))

        self.log_init_lr = torch.nn.Parameter(torch.tensor([-2], device=device, dtype=self.ndtype))
        self.lr_log_tau = torch.nn.Parameter(-torch.log(torch.tensor([.63 * expected_plateu],
                                                                     device=device, dtype=self.ndtype)))
        self.log_final_lr = torch.nn.Parameter(torch.tensor([-2], device=device, dtype=self.ndtype))

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
        self.log_init_lr = torch.nn.Parameter(self.log_init_lr.to(device))
        self.lr_log_tau = torch.nn.Parameter(self.lr_log_tau.to(device))
        self.log_final_lr = torch.nn.Parameter(self.log_final_lr.to(device))
        self.inverse_temperature = torch.nn.Parameter(self.inverse_temperature.to(device))

    def _nonsingular_cov_from_triu(self, vec):
        cov_in = util.triu_to_square(vec, 2, includes_diag=True)
        return cov_in.clone() @ cov_in.clone()

    def fit(self, trial_data: pd.DataFrame, epochs=1000, lr=.01):
        """
        trial_data: DataFrame: cols: s1-s4, c1-c4, reward, choice_idx
        """
        ce = torch.nn.NLLLoss()
        state_spaces = None
        options, choice_idxs, rewards, time_delta = self._get_choices_rewards(trial_data)
        options = torch.round(self.resolution * (options / self.s_real)).int()

        self.optimizer = torch.optim.Adam(params=[self.init_kernel_cov, self.kernel_log_tau,
                                                  self.final_kernel_cov, self.inverse_temperature, self.log_init_lr,
                                                  self.log_final_lr, self.lr_log_tau,
                                                  self.log_init_forget_tau, self.log_final_forget_tau,
                                                  self.log_forget_tau_tau, self.init_prob],
                                          lr=lr)
        for epoch in range(epochs):
            loss = torch.zeros((1,), dtype=self.ndtype, device=self.device)
            self.state_spaces = []
            print("Epoch", epoch)
            self.optimizer.zero_grad()
            # build true nonsingular cov from cov input
            kernel_cov = self._nonsingular_cov_from_triu(self.init_kernel_cov)
            f_kernel_cov = self._nonsingular_cov_from_triu(self.final_kernel_cov)
            # select the input covariance matrix at this time point (tools library)
            kernel_covs = util.exponential_func(torch.arange(len(trial_data)), kernel_cov.flatten(),
                                                f_kernel_cov.flatten(), self.kernel_log_tau).reshape((-1, 2, 2))
            # apply initial kernel to statespace
            kernel = util.gaussian_kernel(kernel_size=(self.resolution, self.resolution),
                                          cov=kernel_cov * self.resolution,
                                          integral_resolution=1,
                                          renormalize=True, )
            pad = ((self.resolution - 1) // 2)
            init_state_space = torch.conv2d(self.init_prob.clone()[None, None, :, :], weight=kernel[None, None, :, :],
                                       padding=pad).squeeze()
            state_space = init_state_space.clone()
            # get all learning rates given learning rate curve
            log_lrs = torch.log(util.exponential_func(torch.arange(len(trial_data)),
                                                      torch.exp(self.log_init_lr),
                                                      torch.exp(self.log_final_lr),
                                                      self.lr_log_tau).flatten())

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
                # compute how much we forgot since the last trial (tools library)
                cur_state = util.exponential_func(torch.tensor([time]), state_space.flatten(),
                                                  init_state_space.clone().flatten(),
                                                  log_forget_tau).reshape(state_space.shape)
                state_space = cur_state

                # construct gaussian kernel (uses spencer's tools library)
                kernel = util.gaussian_kernel(kernel_size=(self.resolution, self.resolution),
                                              cov=kernel_cov * self.resolution,
                                              integral_resolution=1,
                                              renormalize=True, )
                # to (0, 1)
                reward = rewards[ind]
                # get the expected reward of each option from most recent state space
                option_reward_exp = state_space[options[ind, :, 0], options[ind, :, 1]]
                # compute softmax
                choice_probs = torch.log_softmax(option_reward_exp * self.inverse_temperature, dim=0)
                # compare to monkey choice
                choice_idx = choice_idxs[ind]
                loss = loss + ce(choice_probs, choice_idx)
                # add state space to list
                self.state_spaces.append(state_space.clone())
                # # update reward space
                offset_mu = (options[
                                 ind, choice_idx] - self.resolution // 2).detach().cpu().tolist()  # how much to translate kernel
                state_space = self.state_spaces[-1].clone()
                # # center kernel on choice
                centered_kernel = torch.roll(kernel.clone(), offset_mu, dims=(0, 1))
                state_space = state_space + torch.exp(log_lrs[ind]) * centered_kernel * (
                            reward.detach() - state_space)  # kernel need to weight both observed reward and current space
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
        res_dict["init_kernel_cov"] = self._nonsingular_cov_from_triu(
            self.init_kernel_cov).detach().cpu().numpy().reshape((2, 2))
        res_dict["kernel_log_tau"] = self.kernel_log_tau.detach().cpu().numpy()
        res_dict["final_kernel_cov"] = self._nonsingular_cov_from_triu(
            self.final_kernel_cov).detach().cpu().numpy().reshape((2, 2))
        res_dict["init_forget_log_tau"] = self.log_init_forget_tau.detach().cpu().numpy()
        res_dict["final_forget_log_tau"] = self.log_final_forget_tau.detach().cpu().numpy()
        res_dict["forget_log_tau_tau"] = self.log_forget_tau_tau.detach().cpu().numpy()
        res_dict["init_log_learn_rate"] = self.log_init_lr.detach().cpu().numpy()
        res_dict["final_log_learn_rate"] = self.log_final_lr.detach().cpu().numpy()
        res_dict["lr_log_tau"] = self.lr_log_tau.detach().cpu().numpy()
        res_dict["inverse_temperature"] = self.inverse_temperature.detach().cpu().numpy()
        res_dict["initial_value_space"] = self.init_prob.detach().cpu().numpy()
        res_dict["state_spaces"] = [space.detach().cpu().numpy() for space in self.state_spaces]
        return res_dict
