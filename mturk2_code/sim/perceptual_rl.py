import math
import pickle
from typing import List

from mturk2_code.sim.rl_base import BaseRL
import torch
import pandas as pd
import numpy as np
#from neurotools import util
from ProbGraph import ElasticEmbed
import copy
from pybads.bads import BADS
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import multiprocessing as mp
import pickle as pk
from botorch.optim.initializers import draw_sobol_samples
mp.set_start_method('spawn', force=True)


_sin = torch.sin(2 * torch.pi * torch.arange(36) / (36))
_cos = torch.cos(2 * torch.pi * torch.arange(36) / (36))
_c_default_stim = torch.stack([_sin, _cos], dim=1)
_s_defualt_stim = _c_default_stim.clone()


def cartesian(list1, list2):
    list1 = list1.unsqueeze(1)  # (N, 1, d1)
    list2 = list2.unsqueeze(0)  # (1, M, d2)
    prod1 = list1.expand(-1, list2.size(1), -1)  # (N, M, d1)
    prod2 = list2.expand(list1.size(0), -1, -1)  # (N, M, d2)
    out = torch.cat([prod1, prod2], dim=2)       # (N, M, d1 + d2)
    return out.reshape(-1, out.size(2))


def min_max_scale(X, bounds):
    """
    X, tensor <b, params>
    bounds, tensor <params, 2>
    Scale X to [0, 1] given (min, max) bounds.
    """
    return X
    min_vals = bounds[:, 0].unsqueeze(0)
    max_vals = bounds[:, 1].unsqueeze(0)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)


def min_max_unscale(X_scaled, bounds):
    """
    X_scaled, tensor <b, params>
    bounds, tensor <params, 2>
    Invert scaling from [0, 1] back to original domain.
    """
    min_vals = bounds[:, 0].unsqueeze(0)
    max_vals = bounds[:, 1].unsqueeze(0)
    return X_scaled * (max_vals - min_vals + 1e-8) + min_vals


def qei_with_repulsion(model, bounds, best_f, q, candidate_pool=1024):
    """
    Efficient greedy qEI batch selection with distance-based repulsion.

    Args:
        model: A fitted BoTorch GP model.
        bounds: Tensor of shape [2, d], min and max bounds.
        best_f: Best objective value observed.
        q: Number of batch points to select.
        candidate_pool: Number of candidate points to sample.
        min_dist: Minimum L2 distance between batch elements.

    Returns:
        Tensor [q, d] of selected candidates.
    """
    acqf = qLogExpectedImprovement(model=model, best_f=best_f)
    candidates, _ = optimize_acqf(
                    acqf,
                    bounds=bounds,
                    q=q,
                    num_restarts=10,
                    raw_samples=100,
                )
    if q < 5:
        return candidates

    sob_candidates = draw_sobol_samples(bounds=bounds, n=2 * q, q=1, seed=0).squeeze(1)
    candidates = torch.cat([candidates, sob_candidates], dim=0)

    with torch.no_grad():
        scores = acqf(candidates.unsqueeze(1)).squeeze()
    scores = scores - torch.min(scores) + .1

    pairwise_dists = torch.cdist(candidates, candidates)
    min_dist = torch.quantile(pairwise_dists, 1 / q)
    selected_indices = []
    available = torch.ones(len(candidates), dtype=torch.bool)

    for _ in range(q):
        if torch.sum(available) == 0:
            break
        idx = torch.argmax(scores * available.float())
        selected_indices.append(idx.item())
        available[pairwise_dists[idx] < min_dist] = 0

    return candidates[selected_indices]


class CSC_RL(BaseRL):
    def __init__(self, n_c=36, n_s=36, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_history = []
        self.prob_history = []

        # start perceptual distance parameters
        self.rad_c = 5
        self.rad_s = 8.0
        self.rad_min = 5.
        self.rad_max = 25.

        # distance vs value equilibrium coeficciant
        self.theta_1 = 1.0 # distance term
        self.theta_min = 0.
        self.theta_max = 4.

        self.theta_2 = math.log(1000)
        self.bias_min = math.log(1)
        self.bias_max = math.log(10000)

        # value influence power law
        # self.theta_4 = 2.19  # distance floor
        # self.theta_5 = 1.73  # inverse power scale on value term influence
        # self.theta_4_min = .5
        # self.theta_4_max = 3.
        # self.theta_5_min = .1
        # self.theta_5_max = 3.

        # natural log learning rate (value update kernel coefficient)
        self.llr = 5
        self.log_lr_min = 0
        self.log_lr_max = 20

        # temperature (coefficient on softmax probs)
        self.temperature = 3.5
        self.temperature_min = 2.0
        self.temperature_max = 10.

        # True reward updates (mid-rewards yoked to 0 and 1)
        self.low_r = -1.0
        self.low_r_min = -4.
        self.low_r_max = 0.

        self.high_r = 2.0
        self.high_r_min = 1.
        self.high_r_max = 4.

        self.val_init_min = 0.
        self.val_init_max = 1.0
        self.val_init = 1.0
        self.n_colors = n_c
        self.n_shapes = n_s


    def construct_elastic(self, params, track=False, dev="cpu"):
        rad_s, rad_c, theta_1, theta_2, lr, _, rmin, rmax = params
        s_stim_def = rad_s * _s_defualt_stim.to(dev)
        c_stim_def = rad_c * _c_default_stim.to(dev)
        coords = cartesian(c_stim_def, s_stim_def).reshape((-1, 4))
        return ElasticEmbed(4, coords, theta_1=torch.tensor([theta_1]),
                                                    theta_2=torch.tensor([math.exp(theta_2)]),
                                                    reward_override=torch.Tensor([rmin, 0, 1, rmax]),
                                                    lr=torch.exp(lr),
                            compile_opt=True, track=track, dev=dev)

    def episode(self, params, all_options, reward, choices, dev="cuda", track_states=False):
        with torch.no_grad():
            params = [p.to(dev) for p in params]
            all_options = all_options.to(dev)
            reward = reward.to(dev)
            choices = choices.to(dev)
            # pass all params except final three (temp, rmin, rmax)
            model = self.construct_elastic(params, track=track_states, dev=dev)
            temperature = params[-3]
            prob_hist = []
            n_trials = len(all_options)
            log_likelihood = 0.
            for ind in range(n_trials):
                # send index to map to perceptual coordinates (embedding)
                option_ind_c = all_options[ind, 0, :, 0] # <choice>
                option_ind_s = all_options[ind, 0, :, 1]
                # convert to list coord
                options = option_ind_s * self.n_colors + option_ind_c
                # get value for each option under model
                values = model.get_val(options)
                # get choice probs for each option
                lp = torch.log_softmax(values * temperature, dim=0)
                # prob of monkey choice
                choice_idx = choices[ind]
                choice_lp = lp[choice_idx]
                log_likelihood += choice_lp
                if track_states:
                    prob_hist.append(choice_lp.detach().cpu())
                if (ind + 1) % 500 == 0:
                    print("Epi.", ind, "LL", -log_likelihood.detach().cpu().item() / ind)

                # get reward val
                r = reward[ind]

                # apply reward update
                chosen = options[choice_idx].squeeze()
                model.update(int(r), chosen)

                # relax graph
                model.relax()
            print("episode complete")
            if track_states:
                return (torch.stack(model.dist_hist).detach().cpu().numpy(),
                        torch.stack(model.val_hist).detach().cpu().numpy(),
                        torch.stack(prob_hist).detach().cpu().numpy())
            else:
                return -(log_likelihood / n_trials).detach().cpu()

    def prepare(self, trial_datas: List[pd.DataFrame]):
        options = []
        choice_idxs = []
        rewards = []
        # construct behavior data
        for trial_data in trial_datas:
            o, c, r, td = self._get_choices_rewards(trial_data)
            rewards.append(r)
            choice_idxs.append(c)
            options.append(o)
        rewards = torch.stack(rewards, dim=1) - 1 # <t, s>
        options = torch.stack(options, dim=1)  # <t, s, ind, m>
        choice_idxs = torch.stack(choice_idxs, dim=1)  # <t, s>
        return rewards, options, choice_idxs

    def _aq_bach_exec(self, candidates, cls_state):
        # Evaluate objective in parallel across q candidate points
        with ProcessPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(self.episode, c, *cls_state) for c in candidates]
            candidate_vals = torch.cat([f.result() for f in futures]).unsqueeze(1)
        return candidate_vals

    def fit(self, trial_datas: List[pd.DataFrame], *args, **kwargs):
        # Preprocess trial data into tensors
        rewards, options, choice_idxs = self.prepare(trial_datas)


        # Define parameter bounds for the optimizer
        bounds = torch.tensor([
            [self.rad_min, self.rad_max],
            [self.rad_min, self.rad_max],
            [self.theta_min, self.theta_max],
            [self.bias_min, self.bias_max],
            # [self.theta_4_min, self.theta_4_max],
            # [self.theta_5_min, self.theta_5_max],
            [self.log_lr_min, self.log_lr_max],
            [self.temperature_min, self.temperature_max],
            [self.low_r_min, self.low_r_max],
            [self.high_r_min, self.high_r_max],
        ], dtype=torch.float32)

        start = [self.rad_c, self.rad_s, self.theta_1, self.theta_2, self.llr, self.temperature, self.low_r, self.high_r]
        start = torch.tensor(start).unsqueeze(0)
        # Generate initial random design of 10 parameter sets
        init_seq = -1
        init_q = 20
        cls_state = (options[:init_seq], rewards[:init_seq], choice_idxs[:init_seq])
        orig_bounds = bounds.clone()
        train_x = draw_sobol_samples(bounds.T, n=init_q, q=1).squeeze()
        train_x = torch.concat([start, train_x], dim=0)
        train_y = self._aq_bach_exec(train_x, cls_state)
        for i in range(len(train_x)):
            print("X:", train_x[i], "F:", train_y[i])

        train_x = train_x.double()
        train_y = train_y.double()
        # train_x = min_max_scale(train_x, orig_bounds)
        gp_like = []

        # Fit initial GP surrogate model
        gp = SingleTaskGP(train_x, train_y)
        gp.mean_module.batch_shape = 1
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        nll = -torch.mean(mll(gp(train_x), train_y)).detach().item()
        print("FIRST GP-NLL", nll)
        # gp_like.append(nll)

        qs = [5, 5, 5, 3, 3, 3, 3, 3, 3]

        n_stage = 9
        for stage in range(n_stage):
            # Use larger batch size in first stage for exploration
            q = qs[stage]
            num_rounds = 1

            for round_idx in range(num_rounds):

                bounds = torch.zeros(orig_bounds.shape)
                bounds[:, 1] = 1

                # Construct q-expected improvement acquisition function with sobel sampling and min distances
                candidates = qei_with_repulsion(gp, orig_bounds.T, best_f=train_y.min().double().reshape((1)), q=q)


                # construct new bounds, based on previous iters.
                # bounds_max = torch.max(candidates, dim=0)[0] * 1.25
                # bounds_min = torch.min(candidates, dim=0)[0] * .8
                # bounds_max = torch.min(torch.stack([bounds_max, bounds[:, 1]], dim=1), dim=1)[0]
                # bounds_min = torch.max(torch.stack([bounds_min, bounds[:, 0]], dim=1), dim=1)[0]
                # bounds = torch.stack((bounds_min, bounds_max), dim=1)
                # # scale new bounds
                # bounds = min_max_scale(bounds.T, orig_bounds).T

                # SELECT SEQ LEN
                cls_state = (options[:], rewards[:], choice_idxs[:])

                # make inputs safe
                candidates[torch.logical_or(torch.isnan(candidates), torch.isinf(candidates))] = 1e-10

                # evaluate function aq batch
                std_candidates = candidates # min_max_unscale(candidates, orig_bounds)
                candidate_vals = self._aq_bach_exec(std_candidates.float(), cls_state).double()

                # make f vals safe
                candidate_vals[torch.logical_or(torch.isnan(candidate_vals), torch.isinf(candidate_vals))] = 1e-10

                # disp:
                for i in range(len(std_candidates)):
                    print("X:", std_candidates[i], "F:", candidate_vals[i])

                # Append new data to training set
                train_x = torch.cat([train_x, candidates], dim=0)
                train_y = torch.cat([train_y, candidate_vals], dim=0)

                # set saved model params to the best
                best_idx = torch.argmin(train_y)
                best_params = train_x[best_idx].detach().cpu().numpy()
                self.rad_c, self.rad_s, self.theta_1, self.theta_2, self.llr, self.temperature, self.low_r, self.high_r = best_params.tolist()
                print("ITER", stage, "Best NLL:", train_y[best_idx].item())
                print("Best params:", best_params)

                if stage < (n_stage - 1):
                    # Update the model with new data (reuse GP hyperparameters unless refitting)
                    gp.set_train_data(train_x, train_y.squeeze(), strict=False)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    mll = fit_gpytorch_mll(mll)
                    nll = -torch.mean(mll(gp(train_x), train_y)).detach().item()
                    print("STAGE", stage, "GP-NLL", nll)
                    gp_like.append(nll)
        return gp_like

    def predict(self, trial_datas: List[pd.DataFrame], *args, **kwargs):
        rewards, options, choice_idxs = self.prepare(trial_datas)
        start = [self.rad_c, self.rad_s, self.theta_1, self.theta_2, self.llr, self.temperature, self.low_r, self.high_r]
        start = torch.tensor(start)
        self.dist_hist, self.val_hist, self.prob_history = self.episode(start, options, rewards, choice_idxs,
                                                                        track_states=True)
        return self.dist_hist, self.val_hist, self.prob_history

    def get_results(self):
        res = {"color_radius": self.rad_c,
               "shape_radius": self.rad_s,
               "prob_convertion_bias": self.theta_1,
               "assoc_prior_strength": self.theta_2,
               "log_learning_rate": self.llr,
               "temperature": self.temperature,
               "reward_0": self.low_r,
               "reward_1": 0.0,
               "reward_2": 1.0,
               "reward_3": self.high_r}
        return res


class PerceptRL(BaseRL):
    """
    Nonlinearly (using simple MLP) Transforms given stimuli coordinates into estimate of monkey perceptual space. Using
    Rescola-Wagner updates, constructs reward distribution using gaussian kernel with preset covariance.
    """

    def __init__(self, ndims=10, nfeatdims=10, expected_plateu=100000, s_stim_def=_c_default_stim, c_stim_def=_s_defualt_stim,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndims = ndims
        self.nfeatdims = nfeatdims
        self.expected_plateu = expected_plateu
        self.gru_h = torch.nn.GRU(input_size=4 + 1,
                             hidden_size=ndims, bias=True)
        self.pred = torch.nn.Linear(in_features=nfeatdims, out_features=1, bias=True)
        self.state_history = []
        self.prob_history = []
        self.s_stim_def = s_stim_def
        self.c_stime_def = c_stim_def
        self.last_hidden = None

    def predict(self, options, reward, choice_idx=None, use_stored_hidden=True, *args, **kwargs):
        n = len(options)
        n_subjects = options.shape[1]
        n_indices = options.shape[2]
        if reward is None:
            reward = torch.zeros((n, n_subjects))
        reward = torch.tile(reward.unsqueeze(2), (1, n_indices, 1))
        options = options.view((n, n_subjects * n_indices, 2))
        if choice_idx is None:
            choice_idx = torch.zeros(n)

        logits = []
        choice_ind = 0
        for ind in range(n):
            # send index to map to perceptual coordinates (embedding)
            option_vals_c = self.c_stime_def[options[ind, :, 0]].unsqueeze(0)
            option_vals_s = self.s_stim_def[options[ind, :, 1]].unsqueeze(0)
            r = reward[ind].unsqueeze(0)
            # create input sequence of percept embeddings
            option_vals = torch.concat([option_vals_c, option_vals_s, r], dim=2)  # <n, b, f>
            # send seq to grus
            if use_stored_hidden and self.last_hidden is not None:
                hidden = self.last_hidden[:, torch.arange(n_subjects), choice_ind, :].unsqueeze(2)
                hidden = torch.tile(hidden, (1, 1, n_indices, 1))
                hidden = hidden.reshape((1, n_subjects * n_indices, self.ndims))
            else:
                hidden = torch.zeros((1, n_subjects * n_indices, self.ndims))
            h, o = self.gru_h(option_vals, hidden)
            # send output of each trial and each option (on batch) to linear
            trial_logits = self.pred(h.view(n_subjects * n_indices, self.ndims)).view((n_subjects, n_indices))
            logits.append(trial_logits)
            choice_ind = choice_idx[ind]
            o = o.view(1, n_subjects, n_indices, self.ndims)
            self.state_history.append(o.detach().clone())
            self.last_hidden = o
        logits = torch.stack(logits, dim=0) # <n, s, indices>
        return logits

    def fit(self, trial_datas: List[pd.DataFrame], epochs=1000, lr=.01, snap_out=None):
        ce = torch.nn.NLLLoss()
        options = []
        choice_idxs = []
        rewards = []
        for trial_data in trial_datas:
            o, c, r, td = self._get_choices_rewards(trial_data)
            rewards.append(r)
            choice_idxs.append(c)
            options.append(o)
        rewards = torch.stack(rewards, dim=1) # <t, s>
        options = torch.stack(options, dim=1) # <t, s, ind, m>
        choice_idxs = torch.stack(choice_idxs, dim=1) # <t, s>
        self.optimizer = torch.optim.Adam(params=list(self.gru_h.parameters()) + list(self.pred.parameters()), lr=lr)
        nsubj = options.shape[1]
        nind = options.shape[2]
        for epoch in range(epochs):
            if epoch < epochs - 1:
                use_trials = np.random.randint(min(2000, len(rewards) - 1), len(rewards))
            else:
                use_trials = len(rewards)
            o = options[:use_trials]
            c = choice_idxs[:use_trials]
            r = rewards[:use_trials]
            self.last_hidden = None
            self.state_history = []
            logits = self.predict(o, r, c.int()) # <t, s, ind>
            # get probability of choosing each option in batch
            choice_probs = torch.softmax(logits, dim=2).reshape((use_trials * nsubj, nind)) # <t * s, ind>
            # compare to monkey choice via cross entropy
            loss = ce(choice_probs, c.reshape((use_trials * nsubj)))
            # propagate gradient
            loss.backward(retain_graph=False)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_hist.append(loss.detach().cpu().item())
            if snap_out is not None and (epoch + 1) % 100 == 0:
                try:
                    with open(snap_out, "wb") as f:
                        pickle.dump(self, f)
                except Exception:
                    print("Pickling snapshot failed")
            # check early stopping:
            self.optimizer, converged = util.is_converged(self.loss_hist, self.optimizer, 1, epoch)
            if converged:
                print("Stopping Early...")
                break

    def get_results(self):
        res_dict = {}
        res_dict["epoch_loss_history"] = np.array(self.loss_hist)
        return res_dict

    def to(self, dev):
        self.gru_h = self.gru_h.to(dev)
        self.pred = self.pred.to(dev)
        return self

    def clone(self):
        new = PerceptRL(ndims=self.ndims, s_stim_def=self.s_stim_def, c_stim_def=self.c_stime_def)
        new.gru_h = copy.deepcopy(self.gru_h)
        new.pred = copy.deepcopy(self.pred)
        new.last_hidden = self.last_hidden.clone()
        new.loss_hist = self.loss_hist
        return new


class NTPerceptRL(BaseRL):

    def __init__(self, positional_dim=4, s_stim_def=_c_default_stim, c_stim_def=_s_defualt_stim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stim_features = 4
        self.positional_dim = positional_dim
        self.in_features = stim_features + positional_dim
        self.l1 = torch.nn.Linear(in_features=self.in_features, out_features=32)
        self.l2 = torch.nn.Linear(in_features=32, out_features=16)
        self.l3 = torch.nn.Linear(in_features=16, out_features=8)
        self.l4 = torch.nn.Linear(in_features=8, out_features=1)
        self.s_stim_def = s_stim_def
        self.c_stime_def = c_stim_def
        self.activ = torch.nn.LeakyReLU()

    def predict(self, options, times, choice_idx=None, *args, **kwargs):
        n = len(options)
        num_options = options.shape[1]
        assert len(times) == n
        # send index to map to perceptual coordinates (embedding)
        option_vals_c = self.c_stime_def[options[:, :, 0]]
        option_vals_s = self.s_stim_def[options[:, :, 1]]
        pos_encode = torch.tile(util.positional_encode(times, self.positional_dim), (1, num_options, 1))
        # create input sequence of percept embeddings
        option_vals = torch.concat([option_vals_c, option_vals_s, pos_encode], dim=2)  # <n, b, f>
        _real_bs = len(options)
        to_send = option_vals.reshape((_real_bs * num_options, self.in_features))
        h = self.l1(to_send)
        h = self.activ(h* .1)
        h = self.l2(h)
        h = self.activ(h * .1)
        h = self.l3(h)
        h = self.activ(h * .1)
        logits = self.l4(h)
        # repack by each trial
        logits = logits.reshape((_real_bs, num_options))
        return logits

    def fit(self, trial_data: pd.DataFrame, epochs=1000, lr=.01, *args, **kwargs):
        ce = torch.nn.NLLLoss()
        options, choice_idxs, rewards, time_delta = self._get_choices_rewards(trial_data)
        optimizer = torch.optim.Adam(params=list(self.l1.parameters()) +
                                                 list(self.l2.parameters()) +
                                                 list(self.l3.parameters()), lr=lr)

        for epoch in range(epochs):
            self.last_hidden = None
            n = len(options)

            if choice_idxs is None:
                choice_idxs = torch.zeros(n)

            epoch_loss = 0.
            times = torch.arange(n)
            # shuffle trials
            shuf_indexes = torch.from_numpy(np.random.choice(np.arange(n), n, replace=False)).int()
            options = options[shuf_indexes]
            choice_idxs = choice_idxs[shuf_indexes]
            times = times[shuf_indexes]

            # send to linear
            _batch_size = 1000
            for i in range(0, n, _batch_size):
                optimizer.zero_grad()
                batch = options[i:min(n, i + _batch_size)]
                choice_batch = choice_idxs[i:min(n, i + _batch_size)]
                time_batch = times[i:min(n, i + _batch_size)]
                _real_bs = len(batch)
                logits = self.predict(batch, time_batch)
                # compare to monkey choice via cross entropy
                # get probability of choosing each option in batch
                choice_probs = torch.softmax(logits, dim=1)
                loss = ce(choice_probs, choice_batch)
                epoch_loss += loss.detach().cpu().item()
                loss.backward()
                optimizer.step()
            epoch_loss = (epoch_loss * _batch_size) / n
            print("EPOCH", epoch, "LOSS:", epoch_loss)
            self.loss_hist.append(epoch_loss)
            # check early stopping:
            if epoch > 50:
                states = optimizer.state_dict()["state"]
                moments = []
                for p in states.keys():
                    moments.append(states[p]["exp_avg"].flatten())
                moment = torch.abs(torch.concat(moments, dim=0)).mean()
                print(moment)
                if moment < 1e-4:
                    print("Stopping early...")
                    break

    def get_results(self):
        res_dict = {}
        res_dict["epoch_loss_history"] = np.array(self.loss_hist)
        return res_dict

    def to(self, dev):
        self.l1 = self.l1.to(dev)
        self.l2 = self.l2.to(dev)
        self.l3 = self.l3.to(dev)
        self.l4 = self.l4.to(dev)
        return self

    def clone(self):
        new = NTPerceptRL(ndims=self.positional_dim, s_stim_def=self.s_stim_def, c_stim_def=self.c_stime_def)
        new.l1 = copy.deepcopy(self.l1)
        new.l2 = copy.deepcopy(self.l2)
        new.l3 = copy.deepcopy(self.l3)
        new.l4 = copy.deepcopy(self.l4)
        new.loss_hist = self.loss_hist
        return new