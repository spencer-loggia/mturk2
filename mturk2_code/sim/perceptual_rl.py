import pickle
from typing import List

from mturk2_code.sim.rl_base import BaseRL
import torch
import pandas as pd
import numpy as np
from neurotools import util
import copy

_sin = torch.sin(2 * torch.pi * torch.arange(36) / (36))
_cos = torch.cos(2 * torch.pi * torch.arange(36) / (36))
_c_default_stim = torch.stack([_sin, _cos], dim=1)
_s_defualt_stim = _c_default_stim.clone()


class PerceptRL(BaseRL):
    """
    Nonlinearly (using simple MLP) Transforms given stimuli coordinates into estimate of monkey perceptual space. Using
    Rescola-Wagner updates, constructs reward distribution using gaussian kernel with preset covariance.
    """

    def __init__(self, ndims=16, nfeatdims=16, expected_plateu=100000, s_stim_def=_c_default_stim, c_stim_def=_s_defualt_stim,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndims = ndims
        self.nfeatdims = nfeatdims
        self.expected_plateu = expected_plateu
        self.gru_h = torch.nn.GRU(input_size=4 + 1,
                             hidden_size=ndims, bias=True)
        self.pred = torch.nn.Linear(in_features=nfeatdims, out_features=1, bias=True)
        self.state_history = []
        self.s_stim_def = s_stim_def
        self.c_stime_def = c_stim_def
        self.last_hidden = None

    def predict(self, options, reward, choice_idx=None, use_stored_hidden=True, *args, **kwargs):
        n = len(options)
        batch_size = options.shape[1]
        if choice_idx is None:
            choice_idx = torch.zeros(n)

        logits = []
        choice_ind = 0
        for ind in range(n):
            # send index to map to perceptual coordinates (embedding)
            option_vals_c = self.c_stime_def[options[ind, :, 0]].unsqueeze(0)
            option_vals_s = self.s_stim_def[options[ind, :, 1]].unsqueeze(0)
            r = torch.tile(reward[ind].view((1, 1, -1)), (1, batch_size, 1))
            # create input sequence of percept embeddings
            option_vals = torch.concat([option_vals_c, option_vals_s, r], dim=2)  # <n, b, f>
            # send seq to grus
            if use_stored_hidden and self.last_hidden is not None:
                hidden = torch.tile(self.last_hidden[:, choice_ind, :], (1, batch_size, 1))
            else:
                hidden = torch.zeros((1, batch_size, self.ndims))
            h, o = self.gru_h(option_vals, hidden)
            # send output of each trial and each option (on batch) to linear
            trial_logits = self.pred(h.view(-1, self.ndims)[:, :self.nfeatdims]).view((1, batch_size))
            logits.append(trial_logits)
            choice_ind = choice_idx[ind]
            self.state_history.append(o.detach().clone())
            self.last_hidden = o
        logits = torch.concat(logits, dim=0)
        return logits

    def fit(self, trial_data: pd.DataFrame, epochs=1000, lr=.01, snap_out=None):
        ce = torch.nn.NLLLoss()
        options, choice_idxs, rewards, time_delta = self._get_choices_rewards(trial_data)
        self.optimizer = torch.optim.Adam(params=list(self.gru_h.parameters()) + list(self.pred.parameters()), lr=lr)

        for epoch in range(epochs):
            if epoch < epochs - 1:
                use_trials = np.random.randint(800, len(trial_data))
            else:
                use_trials = len(trial_data)
            print(use_trials)
            o = options[:use_trials]
            c = choice_idxs[:use_trials]
            r = rewards[:use_trials]
            td = time_delta[:use_trials]
            self.last_hidden = None
            self.state_history = []
            logits = self.predict(o, r, c.int())
            # get probability of choosing each option in batch
            choice_probs = torch.softmax(logits, dim=1)
            # compare to monkey choice via cross entropy
            loss = ce(choice_probs, c)
            # propagate gradient
            loss.backward(retain_graph=False)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_hist.append(loss.detach().cpu().item())
            if snap_out is not None and (epoch + 1) % 100 == 0:
                try:
                    pickle.dump(self, snap_out)
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