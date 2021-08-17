from typing import List

import pandas as pd
import numpy as np


def present_previous_trials(agent, all_subject_data: List):
    """
    Takes an agent object and a list of session historys (SessionData objects) and returns a new file name and data dict
    that would have resulted from agent preforming the tasks.
    """
    new_data_dict = {
        'Test': [],
        'TestC': [],
        'Response': [],
        'RewardStage': [],
        'StartTime': []
    }
    date = None
    for subject_hist in all_subject_data:
        if not subject_hist:
            # continue if this subject is not in this space group
            continue
        date = subject_hist.date
        new_data_dict['Test'].append(subject_hist.shape_trials)
        new_data_dict['TestC'].append(subject_hist.color_trials)
        new_data_dict['RewardStage'].append(subject_hist.reward_map)
        new_data_dict['StartTime'].append(subject_hist.trial_time_milliseconds)
        choices = []
        for j in range(len(subject_hist.shape_trials)):
            shape_trial = subject_hist.shape_trials[j]
            color_trial = subject_hist.color_trials[j]
            reward_pred = [agent.predict(shape_trial[i], color_trial[i]) for i in range(len(shape_trial))]
            choice = np.argmax(np.array(reward_pred))
            choices.append(choice)
        new_data_dict['Response'].append(np.array(choices))
    for key in new_data_dict.keys():
        new_data_dict[key] = np.concatenate(new_data_dict[key], axis=0)
    group_descriptor = ''.join(filter(None, [s.monkey_name[0] if s else None for s in all_subject_data]))
    file_destrciptor = str(date.year) + '_' + str(date.month) + '_' + str(date.day) + '_x_x_x_' + \
                       str(agent.decision_policy).replace('_', '.') + '.' + group_descriptor + '_sim_' + 'na'
    return file_destrciptor, new_data_dict
