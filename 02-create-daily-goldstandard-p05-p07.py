import os
import pandas as pd
import warnings
import numpy as np
from io import StringIO

import utils
from utils import get_complete_date_range

warnings.filterwarnings('ignore')

for p in [5, 7]:
    root = '/home/ali/PycharmProjects/si/data'
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    to = os.path.join(to, 'p0' + str(p), 'daily')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # GOLDSTANDARD
    modality = 'p0' + str(p) + '-sis'

    timestamps = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si/data', 'p0' + str(p), 'daily', 'step' + '.csv'))
    timestamps['timestamp'] = pd.to_datetime(timestamps['timestamp'])

    goldstandard = pd.read_csv(os.path.join(root, modality + '.csv'))
    goldstandard['timestamp'] = pd.to_datetime(goldstandard['timestamp'])

    i = 0
    # _date = goldstandard[goldstandard.columns[0]][i]
    _date = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i])
    timestamps['target-date-' + str(i)] = timestamps['timestamp'].apply(lambda x: _date if x <= _date else None)

    i = 1
    date_ = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i - 1])
    _date = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i])
    timestamps['target-date-' + str(i)] = timestamps['timestamp'].apply(lambda x: _date if date_ < x <= _date else None)

    i = 2
    date_ = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i - 1])
    _date = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i])
    timestamps['target-date-' + str(i)] = timestamps['timestamp'].apply(lambda x: _date if date_ < x <= _date else None)

    i = 3
    date_ = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i - 1])
    _date = pd.to_datetime(goldstandard[goldstandard.columns[0]].index[i])
    timestamps['target-date-' + str(i)] = timestamps['timestamp'].apply(lambda x: _date if date_ < x else None)

    temp = timestamps['timestamp']

    timestamps = timestamps[[
        'target-date-' + str(0),
        'target-date-' + str(1),
        'target-date-' + str(2),
        'target-date-' + str(3)]]

    timestamps['goldstandard-date'] = timestamps.bfill(axis=1).ffill(axis=1).iloc[:, 0]

    timestamps = pd.concat([
        temp,
        timestamps['goldstandard-date']
    ], axis=1)










    timestamps.to_csv(os.path.join(to, 'goldstandard-date' + '.csv'), index=False)


