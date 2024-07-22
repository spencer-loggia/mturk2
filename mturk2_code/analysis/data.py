import numpy as np
import datetime
import re


class SessionData:
    """
    an object to hold the data from one session
    """

    def __init__(self, data_list):
        """
        :param data_list: list of tuples of fnames and data dictionaries
        """
        base_fname = data_list[0][0]
        items = re.split('-|_', base_fname)
        self.date = datetime.date(int(items[0]), int(items[1]), int(items[2]))
        self.monkey_name = items[6]
        self.mode = items[7]
        self.tablet_used = items[-1]
        self.data_list = data_list
        shape_trials = []
        color_trials = []
        reward_map = []
        choices = []
        resp_xyt = []
        trial_time_milliseconds = []
        for _, data_dict in self.data_list:
            shape_trials.append(np.array(data_dict['Test'], dtype=int))
            reward_map.append(np.array(data_dict['RewardStage'], dtype=int))
            color_trials.append(np.array(data_dict['TestC'], dtype=int))
            choices.append(np.array(data_dict['Response'], dtype=int))
            resp_xyt.append(np.array(data_dict['ResponseXYT'], dtype=float))
            trial_time_milliseconds.append(data_dict['StartTime'])
        self.shape_trials = np.concatenate(shape_trials, axis=0)
        self.color_trials = np.concatenate(color_trials, axis=0)
        self.reward_map = np.concatenate(reward_map, axis=0)
        self.choices = np.concatenate(choices, axis=0)
        self.resp_xyt = np.concatenate(resp_xyt, axis=0)
        self.trial_time_milliseconds = np.concatenate(trial_time_milliseconds, axis=0)

    def get_full_trial(self):
        # s1, s2, s3, s4, c1, c2, c3, c4, reward, choice_idx, respx, respy, react_time
        arr = np.concatenate([self.shape_trials,
                              self.color_trials,
                              self.reward_map[np.arange(len(self.reward_map)), self.choices].reshape(-1, 1),
                              self.choices.reshape(-1, 1),
                              self.resp_xyt[:, :2],
                              (self.resp_xyt[:, 2] - self.trial_time_milliseconds).reshape(-1, 1)], axis=1).astype(int)
        return arr

    def get_priors(self):
        """
        return the observed probability of each reward being presented in this trial.
        """
        prior = [0] * 4
        for i in range(4):
            prior[i] = float(np.count_nonzero(self.reward_map.reshape(-1) == i) / (len(self) * 4))
        return np.array(prior)

    def get_real_reward_dist(self):
        """
        return the observed probability of picking each reward class.
        """
        full_freq = [0] * 4
        dist = np.choose(self.choices, self.reward_map.T)
        item, freq = np.unique(dist, return_counts=True)
        for ind, i in enumerate(item):
            full_freq[i] = freq[ind]
        return full_freq

    def choice_frequency_data(self):
        """
        return  that indicates the number of times a monkey chose a particular color shape pair
        """
        shape_axis = np.choose(self.choices, self.shape_trials.T)
        color_axis = np.choose(self.choices, self.color_trials.T)
        is_best = self.get_max_reward_arr()
        coords = list(zip(shape_axis, color_axis))
        unique_count = {}
        for i, coord in enumerate(coords):
            if coord in unique_count:
                unique_count[coord][0] += 1
            else:
                unique_count[coord] = [1, is_best[i]]
        unique = list(unique_count.keys())
        count, best = list(map(list, zip(*unique_count.values())))  # tranpose and unpack
        return unique, count, best

    def get_max_reward_arr(self):
        best_choice = np.max(self.reward_map, axis=1)
        best_arr = best_choice == np.choose(self.choices, self.reward_map.T)
        return best_arr

    def get_max_reward_prob(self):
        """
        get the observed probability that a monkey picked the best reward on a trial
        """
        num_best = np.count_nonzero(self.get_max_reward_arr())
        return num_best / len(self)

    def get_min_reward_prob(self):
        """
        get the observed probability that a monkey picked the worst reward on a trial
        """
        best_choice = np.min(self.reward_map, axis=1)
        num_best = np.count_nonzero(best_choice == np.choose(self.choices, self.reward_map.T))
        return num_best / len(self)

    def __len__(self):
        return len(self.trial_time_milliseconds)