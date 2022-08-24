import sys
from typing import List, Dict, Union, Tuple, FrozenSet

import dropbox
import numpy as np
from scipy.spatial.distance import pdist, squareform
import json
import os
import re
import datetime
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from io import BytesIO, StringIO
from multiprocessing import Pool

from sim import Agent, ColorShapeData
from data import SessionData
from visualize import heatmap_scatterplot
from connect_task import present_previous_trials

import smtplib
import ssl
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

SUBJECT_NAMES = {'Tina', 'Yuri', 'Sally', 'Buzz'}

HISTORICAL_FEATS = ['date', 'subject_name', 'num_trials', 'duration(last-first)', 'r0_percent_diff_chance',
                    'r1_percent_diff_chance', 'r2_percent_diff_chance', 'r3_percent_diff_chance', 'prob_best_reward',
                    'prob_worst_reward']
VERSION_NOTE = "V11. Added new figure that shows subject choices on map of reward space. Shows trials were best choice " \
               "was made in Green, and trials were a mistake (of any magnitude) was made in cyan"

HISTORICAL_DATE_NOTES = {'2021-09-08': ('A', 'Reward Size Halved (300ms -> 150ms), Penalty Timeout Doubled'),
                         '2021-09-16': ('B', 'Default Inter-Trial Timeouts Added, 2000ms, Max Trial Number Set to 500'),
                         '2021-09-22': ('C', 'Mouthpiece Added to Tina and Yuri Boxes'),
                         '2021-10-05': ('D', 'Inter-Trial Timeout Reduced to 500ms. Reward size reduced to 100ms')}

DESCRIPTION = "Key for Historical plot x labels " + str(HISTORICAL_DATE_NOTES)


def analyze_session(data_list):
    """
    :param data_list: list of tuples of fnames and data dictionaries
    Initialize data analysis and collect results.
    Return a dictionary containing computed statistics
    """
    data = SessionData(data_list)
    observed_r_dist = data.get_real_reward_dist()
    prior_r_dist = data.get_priors()
    analysis = {'observed_reward_dist': observed_r_dist,
                'prior_reward_dist': prior_r_dist * len(data),
                'percent_diff_chance': ((observed_r_dist - (prior_r_dist * len(data))) / (prior_r_dist * len(data))),
                'percent_best_reward': data.get_max_reward_prob(),
                'percent_worst_reward': data.get_min_reward_prob(),
                'choice_freq_data': data.choice_frequency_data()}
    if data.monkey_name in SUBJECT_NAMES:
        analysis['xy_coords'] = data.resp_xyt[:, :2]
    else:
        analysis['xy_coords'] = None

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
    subject_data = {}
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
                # name_set.remove(sname)
                metadata, fdata = dbx.files_download(
                    path='/Apps/ShapeColorSpace/MonkData/' + f.name)
                if sname in subject_data:
                    subject_data[sname].append((f.name, json.loads(fdata.content)[0]))
                else:
                    subject_data[sname] = [(f.name, json.loads(fdata.content)[0])]
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
            historical_data[name] = pd.read_csv(BytesIO(fdata.content)).sort_values(by='date', axis=0)
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


def handler(subject_data: Dict[str, List[Tuple[str, dict]]],
            historical_data: Dict[str, pd.DataFrame],
            sim_extra):
    """
    Collects analysis, generates test output, and plots analysis.
    returns output text, and saves figures to the saved_data/figures directory
    """
    out = ''
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots(len(subject_data))

    fig3.set_size_inches(8, 6 * (len(subject_data) + 1))

    color_map = cm.get_cmap('plasma')
    exp_name_set = copy.deepcopy(SUBJECT_NAMES)
    res = [analyze_session(s) for s in subject_data.values()]
    sim_res = []

    # the below (and the whole handler function really) a bit wonky, new unexpected features have caused data flow
    # to become overly complex. Should rework project architecture, ideally when full set of desired capabilities
    # is known
    for key in list(sim_extra.keys()):
        space_group_data = [d[1] if d[1].monkey_name in key else None for d in res]
        for a in sim_extra[key]['agents']:
            sim_res.append(
                analyze_session(
                    present_previous_trials(a,
                                            space_group_data)))
            sim_extra[frozenset({sim_res[-1][1].monkey_name})] = {'space_params': sim_extra[key]['space_params']}

    res = res + sim_res
    fig4, ax4 = plt.subplots(len(res))
    fig4.set_size_inches(8, 4 * (len(res) + 1))
    z_range = [None, None]
    all_date = set()
    for i, s in enumerate(res):
        analysis = s[0]
        data = s[1]
        try:
            exp_name_set.remove(data.monkey_name)
        except KeyError:
            pass
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
        try:
            historical_data[data.monkey_name] = pd.concat([historical_data[data.monkey_name], new_row],
                                                          ignore_index=True)
        except KeyError:
            # key error can be thrown due to simulated agents
            pass

        reward_param = None
        for key in sim_extra.keys():
            if data.monkey_name in key:
                reward_param = sim_extra[key]['space_params'].rewards.reshape(36, 36).T

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
        ax1.plot(np.arange(4), analysis['percent_diff_chance'].reshape(-1) * 100,
                 label=data.monkey_name,
                 linestyle='--',
                 marker='o')
        ax1.set_xticks([0, 1, 2, 3])
        ax1.set_xlabel("Reward Type (Worst to Best)")
        ax1.set_ylabel("Reward Frequency Percent Difference From Chance ")
        ax1.set_title("MTurk2 Subject Performance " + str(data.date))
        fig1.legend(loc=(.127, .55))

        if reward_param is not None:
            heatmap_scatterplot(ax4[i], reward_param,
                                         analysis['choice_freq_data'][0],
                                         analysis['choice_freq_data'][1],
                                         analysis['choice_freq_data'][2])
            ax4[i].set_xlabel("Shape Axis")
            ax4[i].set_ylabel("Color Axis")
            ax4[i].set_title("MTurk2 Subject Choices on Reward Map " + data.monkey_name)
        if data.monkey_name in SUBJECT_NAMES:
            dates = [str(ind) for ind in historical_data[data.monkey_name]['date']]
            xlabel = copy.deepcopy(dates)
            all_date = all_date | set(dates)
            ax2.plot(xlabel,
                     historical_data[data.monkey_name]['prob_best_reward'],
                     label=data.monkey_name,
                     linestyle='--',
                     marker='o')
            ax2.set_ylabel("Probability of Choosing Best Reward on Each Trial")
            ax2.set_title("MTurk2 Historical Performance " + str(data.date))
            fig2.legend(loc='upper right')

            x = analysis['xy_coords']
            z = 1 / (squareform(pdist(x)).mean(axis=1))
            minz = min(z)
            maxz = max(z)
            if not z_range[0] or minz < z_range[0]:
                z_range[0] = minz
            if not z_range[1] or maxz > z_range[0]:
                z_range[1] = maxz
            ax3[i].scatter(x[:, 0], x[:, 1], c=z, cmap=color_map)
            ax3[i].set_title(data.monkey_name + ' Choice Location Density Map ' + str(data.date))
            ax3[i].set(adjustable='box', aspect='equal')
    for unused_name in exp_name_set:
        new_row = pd.Series()
        new_row['date'] = str(date)
        historical_data[unused_name] = historical_data[unused_name].append(new_row, ignore_index=True)

    if len(subject_data) > 0:
        all_date = sorted(list(all_date))
        xtick_locs = [j for j, d in enumerate(all_date) if d in HISTORICAL_DATE_NOTES]
        ax2.xaxis.set_ticks(xtick_locs)
        fig2.autofmt_xdate()
        fig1.savefig('../saved_data/figures/' + str(date) + '_performance_vs_chance_mturk2.png')
        fig2.savefig('../saved_data/figures/' + str(date) + '_historical_mturk2.png')
        scalarmappaple3 = cm.ScalarMappable(cmap=color_map)
        scalarmappaple3.set_array(np.arange(z_range[0], z_range[1], (z_range[1] - z_range[0]) / 10))
        fig3.colorbar(scalarmappaple3, orientation='horizontal')
        fig3.savefig('../saved_data/figures/' + str(date) + '_choice_loc_density_mturk2.png')
        color_map = cm.get_cmap('hot')
        scalarmappaple4 = cm.ScalarMappable(cmap=color_map)
        scalarmappaple4.set_array(np.arange(4))
        fig4.colorbar(scalarmappaple4, orientation='horizontal')
        fig4.savefig('../saved_data/figures/' + str(date) + '_choice_reward_map_mturk2.jpg')

    return out, historical_data


def _attach_image(fname: str, m_message: MIMEMultipart):
    with open('../saved_data/figures/' + fname, 'rb') as attach:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attach.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {fname}"
    )
    m_message.attach(part)
    return m_message


def communicate(psswd, ptext, date, debug=False):
    """
    Constructs and sends an email to the default list of recipients.
    If debug is true, only sends to spencer.
    Attaches generated plots to email body.
    """
    if debug:
        to = ["spencer.loggia@nih.gov"]
    else:
        to = ["spencer.loggia@nih.gov",
              "sl@spencerloggia.com",
              "bevil.conway@nih.gov",
              "tunktunk@icloud.com",
              "stuart.duffield@nih.gov"]

    ptext = ptext + '\n\nNote\n' + DESCRIPTION + '\n\nChangeLog\n' + VERSION_NOTE

    m_message = MIMEMultipart("alternative")
    m_message['Subject'] = ptext.partition('\n')[0]
    m_message['From'] = "mturk2mailserverSL@gmail.com"
    m_message['To'] = str(to)

    img_name = str(date) + '_performance_vs_chance_mturk2.png'
    m_message = _attach_image(img_name, m_message)

    img_name = str(date) + '_historical_mturk2.png'
    m_message = _attach_image(img_name, m_message)

    img_name = str(date) + '_choice_loc_density_mturk2.png'
    m_message = _attach_image(img_name, m_message)

    img_name = str(date) + '_choice_reward_map_mturk2.jpg'
    m_message = _attach_image(img_name, m_message)

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
        raise ValueError('Mode must be --prod, --test, --test_save_hist, or --test_mail, not ' + str(mode))
    dbx = dropbox_connector(dbx_token)
    subject_data = get_session_data(dbx, date)
    gt_data_ty = ColorShapeData('../../data/images/imp0.png',
                                '../../data/reward_space_TY.csv',
                                '../../data/freq_space_TY.csv',
                                samples=36 * 36)
    gt_data_sb = ColorShapeData('../../data/images/imp0.png',
                                '../../data/reward_space_SB.csv',
                                '../../data/freq_space_SB.csv',
                                samples=36 * 36)
    ty_space_group = {'Tina', 'Yuri'}
    sb_space_group = {'Sally', 'Buzz'}
    bli_agent_ty = Agent(36, 36, decision_policy='bli')
    bli_agent_ty.fit(gt_data_ty)
    bli_agent_sb = Agent(36, 36, decision_policy='bli')
    bli_agent_sb.fit(gt_data_sb)
    omni_agent_ty = Agent(36, 36, decision_policy='optimal')
    omni_agent_ty.fit(gt_data_ty)
    omni_agent_sb = Agent(36, 36, decision_policy='optimal')
    omni_agent_sb.fit(gt_data_sb)

    output = 'MTurk 2 Progress Report for ' + str(date) + '\n'
    if len(subject_data) == 0:
        output += 'MTurk2 boxes were not setup today.'

    historical = get_historical_data(dbx)

    res = handler(subject_data, historical, {frozenset(ty_space_group): {'agents': [bli_agent_ty, omni_agent_ty],
                                                                         'space_params': gt_data_ty},
                                             frozenset(sb_space_group): {'agents': [bli_agent_sb, omni_agent_sb],
                                                                         'space_params': gt_data_sb}
                                             })
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
