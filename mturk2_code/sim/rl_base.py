import torch
import pandas as pd
import numpy as np
from neurotools import util


class BaseRL:
    def __init__(self, precision="single", device="cpu", *args, **kwargs):
        self.device = device
        if precision == "single":
            self.ndtype = torch.float32
        elif precision == "double":
            self.ndtype = torch.float64
        else:
            raise ValueError
        self.s_real = 36
        self.c_real = 36
        self.loss_hist = []

    def fit(self, trial_data: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError

    def get_results(self):
        raise NotImplementedError

    def _get_choices_rewards(self, trial_data):
        """
        <n, 4, 2> choices on range (0, s_real | c_real)
        """
        raw_choices = trial_data[["c1", "s1", "c2", "s2", "c3", "s3", "c4", "s4"]].to_numpy().reshape((-1, 4, 2))
        time_stamps = trial_data[["time_delta_sec"]].to_numpy().flatten()
        time_stamps = time_stamps - np.concatenate(
            [time_stamps[0][None], time_stamps[:-1]])  # every time stamp minus the one before
        time_stamps = time_stamps / (60 * 60)  # convert from seconds to hours.
        raw_choices = torch.from_numpy(raw_choices).to(int).to(self.device)
        choice_idx = torch.from_numpy(trial_data.choice_idx.to_numpy()).long().to(self.device)
        reward = torch.from_numpy(trial_data.reward.to_numpy()).to(self.ndtype).to(self.device)
        return raw_choices.int(), choice_idx, reward, time_stamps