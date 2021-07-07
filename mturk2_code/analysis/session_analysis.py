import sys
from typing import List, Dict, Union, Tuple

import dropbox
import numpy as np
import json
import os
import re
import datetime
import copy
from scipy import stats

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
        self.data_dict =data_dict
        self.shape_trials = np.array(self.data_dict['Test'], dtype=int)
        self.color_trials = np.array(self.data_dict['TestC'], dtype=int)
        self.reward_map = np.array(self.data_dict['RewardStage'], dtype=int)
        self.choices = np.array(self.data_dict['Response'], dtype=int)
        self.trial_time_microseconds = self.data_dict['StartTime']

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

    def __len__(self):
        return len(self.data_dict['StartTime'])


def analyze_session(filename, data_dict):
    data = SessionData(filename, data_dict)
    observed_r_dist = data.get_real_reward_dist()
    prior_r_dist = data.get_priors()
    out = ''
    out += "Subject: " + str(data.monkey_name) + '\n\n'
    out += "Session Date: " + str(data.date) + '\n\n'
    out += "Trials Completed: " + str(len(data)) + '\n\n'
    out += "Session Runtime: " + str((data.trial_time_microseconds[-1] / 1e3) / (60*60)) + " hours" + '\n\n'
    out += "Frequency Subject Received Each Reward Type (Worst to Best): " + str(list(observed_r_dist)) + '\n\n'
    out += "Chance Frequency of Each Reward Type (Worst to Best): " + str(list(prior_r_dist * len(data))) + '\n\n'
    return out

def dropbox_connector():
    with dropbox.Dropbox(oauth2_access_token='rs2UxmS43BwAAAAAAAAAAXfNzkhWFgemApwtub7zhgWix78XTKrNQCQnyFOgE7zt') as dbx:
        try:
            print(dbx.users_get_current_account())
        except Exception:
            print('Could Not Access Dropbox Using Provided OAuth Token.')
            exit(1)
        return dbx


def get_session_data(dbx, date: datetime.date) -> List[Tuple[str, Dict]]:
    date_str = date.strftime("%Y-%m-%d")
    subject_data = []
    for n in SUBJECT_NAMES:
        args = dropbox.files.SearchOptions(path='/Apps/ShapeColorSpace/MonkData',
                                           max_results=10)
        res = dbx.files_search_v2(query=date_str + ' ' + n, options=args)
        for r in res.matches:
            if r.metadata._value.size > 100 and n in r.metadata._value.name:
                metadata, fdata = dbx.files_download(
                    path='/Apps/ShapeColorSpace/MonkData/' + r.metadata._value.name)
                subject_data.append((r.metadata._value.name, json.loads(fdata.content)[0]))
                break
    print('Loaded File Data Successfully')
    return subject_data


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
                                  "stuart.duffield@nih.gov",
                                  "shriya.awasthi@nih.gov",
                                  "bevil.conway@nih.gov",
                                  "tunk.tunk@icloud.com"],
                        msg=message,)
        print("Report Email Delivered Successfully.")


if __name__=='__main__':
    dbx = dropbox_connector()
    try:
        date = datetime.date(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    except (IndexError, ValueError):
        date = datetime.date.today()
    subject_data = get_session_data(dbx, date)
    output = 'MTurk 2 Progress Report for ' + str(date) + '\n'
    if len(subject_data) == 0:
        output += 'MTurk2 boxes were not setup today.'
    for s in subject_data:
        output += '---------------------------------------\n'
        output += analyze_session(s[0], s[1])
    print(output)
    communicate(sys.argv[1], output)
