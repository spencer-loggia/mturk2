import sys
from typing import List, Dict, Union, Tuple

import dropbox
import numpy as np
import json
import os
import re
import datetime
import copy
from matplotlib import pyplot as plt
import pandas as pd

SUBJECT_NAMES = {'Buzz', 'Tina', 'Yuri', 'Sally'}
VERSION_NOTE = "The above is preliminary analysis, more detailed / interesting information will be added."

class SessionData:
    """
    an object to hold the data from one session
    """

    def __init__(self, base_fname, data_dict):
        items = re.split('-|_', base_fname)
        self.date = datetime.date(int(items[0]), int(items[1]), int(items[2]))
        self.monkey_name = items[6]
        self.mode = items[7]
        self.tablet_used = items[-1]
        self.data_dict = data_dict
        self.shape_trials = np.array(self.data_dict['Test'], dtype=int)
        self.color_trials = np.array(self.data_dict['TestC'], dtype=int)
        self.reward_map = np.array(self.data_dict['RewardStage'], dtype=int)
        self.choices = np.array(self.data_dict['Response'], dtype=int)
        self.trial_time_milliseconds = self.data_dict['StartTime']


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
        dist = np.choose(self.choices, self.reward_map.T)
        freq = np.unique(dist, return_counts=True)[1]
        return freq

    def get_max_reward_prob(self):
        """
        get the observed probability that a monkey picked the best reward on a trial
        """
        best_choice = np.max(self.reward_map, axis=1)
        num_best = np.count_nonzero(best_choice == np.choose(self.choices, self.reward_map.T))
        return num_best / len(self)

    def __len__(self):
        return len(self.data_dict['StartTime'])


def analyze_session(filename, data_dict):
    """
    Initialize data analysis and collect results.
    Return text descripters at index 0, with further indices being data for plotting
    Store any historical data to csv.
    """
    data = SessionData(filename, data_dict)
    observed_r_dist = data.get_real_reward_dist()
    prior_r_dist = data.get_priors()
    analysis = {'observed_reward_dist': observed_r_dist,
                'prior_reward_dist': prior_r_dist * len(data),
                'percent_diff_chance': ((observed_r_dist - (prior_r_dist  * len(data))) / (prior_r_dist * len(data))),
                'percent_best_reward': data.get_max_reward_prob()}
    return analysis, data

def dropbox_connector(access_token: str):
    with dropbox.Dropbox(oauth2_access_token='rs2UxmS43BwAAAAAAAAAAXfNzkhWFgemApwtub7zhgWix78XTKrNQCQnyFOgE7zt') as dbx:
        try:
            print(dbx.users_get_current_account())
        except Exception:
            print('Could Not Access Dropbox Using Provided OAuth Token.')
            exit(1)
        return dbx


def get_session_data(dbx, date: datetime.date) -> List[Tuple[str, Dict]]:
    name_set = copy.deepcopy(SUBJECT_NAMES)
    subject_data = []
    res = dbx.files_list_folder('/Apps/ShapeColorSpace/MonkData/')
    complete = False
    while not complete:
        for f in res.entries:
            if len(name_set) == 0:
                break
            items = re.split('-|_', f.name)
            try:
                sname = items[6]
            except IndexError:
                continue
            if f.size > 0 and sname in name_set:
                reported_datetime = f.client_modified
                if reported_datetime.year != date.year or reported_datetime.month != date.month or reported_datetime.day != date.day:
                    continue
                name_set.remove(sname)
                metadata, fdata = dbx.files_download(
                    path='/Apps/ShapeColorSpace/MonkData/' + f.name)
                subject_data.append((f.name, json.loads(fdata.content)[0]))
        if res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
        else:
            complete = True

    print('Loaded File Data Successfully')
    return subject_data

def handler(subject_data: list):
    out = ''
    colors = ['red', 'green', 'blue', 'purple']
    for i, s in enumerate(subject_data):
        analysis, data = analyze_session(s[0], s[1])
        out += '---------------------------------------\n'
        out += "Subject: " + str(data.monkey_name) + '\n\n'
        out += "Session Date: " + str(data.date) + '\n\n'
        out += "Trials Completed: " + str(len(data)) + '\n\n'
        out += "Session Runtime: " + str((data.trial_time_milliseconds[-1] / 1e3) / (60 * 60)) + " hours" + '\n\n'
        out += "Frequency Subject Received Each Reward Type (Worst to Best): " + str(list(analysis['observed_reward_dist'])) + '\n\n'
        out += "Chance Frequency of Each Reward Type (Worst to Best): " + str(list(analysis['prior_reward_dist'])) + '\n\n'
        out += "Percent Difference of Observed vs Chance: " + str(list(analysis['percent_diff_chance'] * 100)) + "\n\n"
        out += "Observed Portion of Trials Subject Chose Best Available Reward: " + str(analysis['percent_best_reward']) + '\n\n'
        plt.plot(np.arange(4), analysis['percent_diff_chance'].reshape(-1),
                 color=colors[i],
                 label=data.monkey_name,
                 linestyle='--',
                 marker='o')
        plt.xlabel("Reward Type (Worst to Best)")
        plt.ylabel("Reward Frequency Percent Difference From Chance ")
        plt.title("MTurk2 Subject Performance " + str(data.date))
        plt.legend()
    plt.suptitle('')
    plt.savefig('../saved_data/figures/' + str(data.date) + '_performance_vs_chance_mturk2.png')
    return out



def communicate(psswd, message):
    import smtplib
    import ssl
    port = 465
    context = ssl.create_default_context()
    message = 'Subject:' + message + '\n\n' + VERSION_NOTE
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("mturk2mailserverSL@gmail.com", psswd)
        server.sendmail(from_addr="mturk2mailserverSL@gmail.com",
                        to_addrs=["spencer.loggia@nih.gov",
                                  ],
                        msg=message,)
        print("Report Email Delivered Successfully.")


if __name__=='__main__':
    try:
        date = datetime.date(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        psswd = sys.argv[4]
        dbx_token = sys.argv[5]
        mode = sys.argv[6]
    except (IndexError, ValueError):
        date = datetime.date.today()
        psswd = sys.argv[1]
        dbx_token = sys.argv[2]
        mode = sys.argv[3]
    if mode not in ['--prod', '--test']:
        raise ValueError('Mode must be --prod or --test.')
    dbx = dropbox_connector(dbx_token)
    subject_data = get_session_data(dbx, date)
    output = 'MTurk 2 Progress Report for ' + str(date) + '\n'
    if len(subject_data) == 0:
        output += 'MTurk2 boxes were not setup today.'
    output += handler(subject_data)
    print(output)
    if mode == '--prod':
        communicate(psswd, output)
