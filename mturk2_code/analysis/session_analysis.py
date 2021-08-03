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
from io import BytesIO, StringIO
from multiprocessing import Pool


SUBJECT_NAMES = {'Buzz', 'Tina', 'Yuri', 'Sally'}
HISTORICAL_FEATS = ['date', 'subject_name', 'num_trials', 'duration(last-first)', 'r0_percent_diff_chance', 'r1_percent_diff_chance', 'r2_percent_diff_chance', 'r3_percent_diff_chance', 'prob_best_reward', 'prob_worst_reward']
VERSION_NOTE = "V3. Added performance history (see second  attached) and some more stats."


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

    def reward_choice_frequency_marginal(self, axis: str):
        """
        return the vector of the frequency that an item presented was chosen best by subject
        param: axis:str exists in {'shape', 'color'}
        returns: np.ndarray (1xn)
        """
        raise NotImplemented

    def choice_loc_marginal(self):
        """
        return the frequency vector at which each screen location is chosen
        return: np.ndarray (1xd)
        """
        raise NotImplemented

    def get_max_reward_prob(self):
        """
        get the observed probability that a monkey picked the best reward on a trial
        """
        best_choice = np.max(self.reward_map, axis=1)
        num_best = np.count_nonzero(best_choice == np.choose(self.choices, self.reward_map.T))
        return num_best / len(self)

    def get_min_reward_prob(self):
        """
        get the observed probability that a monkey picked the worst reward on a trial
        """
        best_choice = np.min(self.reward_map, axis=1)
        num_best = np.count_nonzero(best_choice == np.choose(self.choices, self.reward_map.T))
        return num_best / len(self)

    def __len__(self):
        return len(self.data_dict['StartTime'])


def analyze_session(filename, data_dict):
    """
    Initialize data analysis and collect results.
    Return a dictionary containing computed statistics
    """
    data = SessionData(filename, data_dict)
    observed_r_dist = data.get_real_reward_dist()
    prior_r_dist = data.get_priors()
    analysis = {'observed_reward_dist': observed_r_dist,
                'prior_reward_dist': prior_r_dist * len(data),
                'percent_diff_chance': ((observed_r_dist - (prior_r_dist * len(data))) / (prior_r_dist * len(data))),
                'percent_best_reward': data.get_max_reward_prob(),
                'percent_worst_reward': data.get_min_reward_prob()}
    return analysis, data


def dropbox_connector(access_token: str):
    with dropbox.Dropbox(oauth2_access_token=access_token) as dbx:
        try:
            print(dbx.users_get_current_account())
        except Exception:
            print('Could Not Access Dropbox Using Provided OAuth Token.')
            exit(1)
        return dbx


def get_session_data(dbx, date: datetime.date) -> List[Tuple[str, Dict]]:
    """
    Scans dropbox for the specified date's data files, and parses them using pythons json module.
    Returns a list of dictionaries of data.
    """
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


def get_historical_data(dbx) -> Dict[str, pd.DataFrame]:
    historical_data = {}
    for name in SUBJECT_NAMES:
        try:
            metadata, fdata = dbx.files_download(path='/Apps/ShapeColorSpace/MonkData/mturk2_' + name + '_history.csv')
            historical_data[name] = pd.read_csv(BytesIO(fdata.content))
        except:
            historical_data[name] = pd.DataFrame(columns=HISTORICAL_FEATS)
    return historical_data


def save_historical_data(dbx, historical_data: pd.DataFrame):
    """
    write new historical data to dropbox
    """
    name = str(historical_data['subject_name'].iloc[0])
    path = '/Apps/ShapeColorSpace/MonkData/mturk2_' + name + '_history.csv'
    try:
        dbx.files_upload(bytes(historical_data.to_csv(), encoding='utf-8'),
                         path=path,
                         mute=True,
                         mode=dropbox.files.WriteMode('overwrite'))
    except Exception:
        raise RuntimeError()
    return


def handler(subject_data: list, historical_data: Dict[str, pd.DataFrame]):
    """
    Collects analysis, generates test output, and plots analysis.
    returns output text, and saves figures to the saved_data/figures directory
    """
    out = ''
    colors = ['red', 'green', 'blue', 'purple']
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i, s in enumerate(subject_data):
        analysis, data = analyze_session(s[0], s[1])
        sess_duration = (data.trial_time_milliseconds[-1] / 1e3) / (60 * 60)
        new_row = pd.DataFrame.from_dict(
            {9999999: [data.date,
                data.monkey_name,
                len(data),
                sess_duration,
                float(analysis['percent_diff_chance'][0]),
                float(analysis['percent_diff_chance'][1]),
                (analysis['percent_diff_chance'][2]),
                float(analysis['percent_diff_chance'][3]),
                float(analysis['percent_best_reward']),
                float(analysis['percent_worst_reward'])
                ]}, orient='index', columns=HISTORICAL_FEATS)
        historical_data[data.monkey_name] = pd.concat([historical_data[data.monkey_name], new_row], ignore_index=True)
        out += '---------------------------------------\n'
        out += "Subject: " + str(data.monkey_name) + '\n\n'
        out += "Session Date: " + str(data.date) + '\n\n'
        out += "Trials Completed: " + str(len(data)) + '\n\n'
        out += "Session Runtime: " + str(sess_duration) + " hours" + '\n\n'
        out += "Frequency Subject Received Each Reward Type (Worst to Best): " + str(
            list(analysis['observed_reward_dist'])) + '\n\n'
        out += "Chance Frequency of Each Reward Type (Worst to Best): " + str(
            list(analysis['prior_reward_dist'])) + '\n\n'
        out += "Percent Difference of Observed vs Chance: " + str(list(analysis['percent_diff_chance'] * 100)) + "\n\n"
        out += "Observed Portion of Trials Subject Chose Best Available Reward: " + str(
            analysis['percent_best_reward']) + '\n\n'
        ax1.plot(np.arange(4), analysis['percent_diff_chance'].reshape(-1),
                 color=colors[i],
                 label=data.monkey_name,
                 linestyle='--',
                 marker='o')
        ax1.set_xticks([0, 1, 2, 3])
        ax1.set_xlabel("Reward Type (Worst to Best)")
        ax1.set_ylabel("Reward Frequency Percent Difference From Chance ")
        ax1.set_title("MTurk2 Subject Performance " + str(data.date))
        fig1.legend()
        ax2.plot([str(ind) for ind in historical_data[data.monkey_name]['date']], historical_data[data.monkey_name]['prob_best_reward'],
                 color=colors[i],
                 label=data.monkey_name,
                 linestyle='--',
                 marker='o')
        ax2.set_xlabel("Session Date")
        ax2.set_ylabel("Probability of Choosing Best Reward on Each Trial")
        ax2.set_title("MTurk2 Historical Performance " + str(data.date))
        fig2.legend()
    if len(subject_data) > 0:
        fig1.savefig('../saved_data/figures/' + str(data.date) + '_performance_vs_chance_mturk2.png')
        fig2.savefig('../saved_data/figures/' + str(data.date) + '_historical_mturk2.png')
    return out, historical_data


def communicate(psswd, ptext, date, debug=False):
    """
    Constructs and sends an email to the default list of recipients.
    If debug is true, only sends to spencer.
    Attaches generated plots to email body.
    """
    import smtplib
    import ssl
    from email.mime.text import MIMEText
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart

    if debug:
        to = ["spencer.loggia@nih.gov"]
    else:
        to = ["spencer.loggia@nih.gov",
              "sl@spencerloggia.com",
              "bevil.conway@nih.gov",
              "tunktunk@icloud.com",
              "stuart.duffield@nih.gov",
              "shriya.awasthi@nih.gov"]

    ptext = ptext + '\n\n' + VERSION_NOTE

    m_message = MIMEMultipart("alternative")
    m_message['Subject'] = ptext.partition('\n')[0]
    m_message['From'] = "mturk2mailserverSL@gmail.com"
    m_message['To'] = str(to)

    img_name = str(date) + '_performance_vs_chance_mturk2.png'
    with open('../saved_data/figures/' + img_name, 'rb') as attach:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attach.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {img_name}"
    )
    m_message.attach(part)

    img_name = str(date) + '_historical_mturk2.png'
    with open('../saved_data/figures/' + img_name, 'rb') as attach:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attach.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {img_name}"
    )
    m_message.attach(part)

    m_message.attach(MIMEText(ptext, "plain"))
    etext = m_message.as_string()

    port = 465
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("mturk2mailserverSL@gmail.com", psswd)
        server.sendmail(from_addr="mturk2mailserverSL@gmail.com",
                        to_addrs=to,
                        msg=etext, )
        print("Report Email Delivered Successfully.")


if __name__ == '__main__':
    try:
        date = datetime.date(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        psswd = sys.argv[4]
        dbx_token = sys.argv[5]
        mode = sys.argv[6]
    except (IndexError, ValueError):
        date = datetime.date.today() - datetime.timedelta(days=1)
        psswd = sys.argv[1]
        dbx_token = sys.argv[2]
        mode = sys.argv[3]
    if mode not in ['--prod', '--test', '--test_mail', '--test_save_hist']:
        raise ValueError('Mode must be --prod, --test, or --test_mail.')
    dbx = dropbox_connector(dbx_token)
    subject_data = get_session_data(dbx, date)
    output = 'MTurk 2 Progress Report for ' + str(date) + '\n'
    if len(subject_data) == 0:
        output += 'MTurk2 boxes were not setup today.'

    historical = get_historical_data(dbx)
    res = handler(subject_data, historical)
    output += res[0]
    hist = res[1]
    if mode in ['--prod', '--test_save_hist']:
        with Pool(len(historical)) as p:
            p.starmap(save_historical_data, [(dbx, h) for h in historical.values()])
    print(output)
    if mode == '--prod':
        communicate(psswd, output, date, debug=False)
    if mode == '--test_mail':
        communicate(psswd, output, date, debug=True)
