import os
import warnings

import pandas as pd

from utils import get_complete_date_range

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:

    root = '/home/ali/PycharmProjects/si/data'
    # root = os.path.join(root, 'p0' + str(p), 'raw')
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    # to = os.path.join(to, 'p0' + str(p), 'daily')
    to = os.path.join(to, 'p0' + str(p), 'daily')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # SLEEP
    modality = 'sleep'
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.rename(columns={'datetime': 'timestamp'}, inplace=True)

    freq = 'D'

    _df_ = df

    _df_ = _df_.reindex(get_complete_date_range(_df_, freq))
    _df_.index.name = 'timestamp'
    _df_.fillna(0, inplace=True)
    _df_.reset_index(inplace=True)

    _df_.to_csv(os.path.join(to, modality + '.csv'), index=False)
    print(p, modality, '\t', df.shape, _df_.shape)


