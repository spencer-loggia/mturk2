# Extracting dropbox files
## Improvised from Spencer's analysis code

import dropbox
import numpy as np
import pandas as pd
import json
import os
import re
import datetime
import copy

from mturk2_code.analysis.session_analysis import dropbox_connector, get_session_data

SUBJECT_NAMES = {'Tina', 'Yuri', 'Sally', 'Buzz'}

#%% Functions taken from Spencer's code to extract files from dropbox
#Don't copy the functions just use them from wherever

session_dates = pd.date_range(start = '2022-08-24', end = '2023-03-06')
session_dates_index = session_dates.strftime("%Y %m %d")

dropbox_token = "KQYKtfoNe5QAAAAAAAAAAXw1PlXCHN99sq0k75mvx1xhA-UXOhVpnjE6NIMGZ7ti"
dbx = dropbox_connector(dropbox_token)

new_data = get_session_data(dbx, '2023 03 06')

