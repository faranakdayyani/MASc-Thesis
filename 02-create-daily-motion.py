import os
import pandas as pd
import warnings
import numpy as np

import utils
from utils import get_complete_date_range

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:

    root = '/home/ali/PycharmProjects/si/data'
    # root = os.path.join(root, 'p0' + str(p), 'hourly')
    root = os.path.join(root, 'p0' + str(p), 'hourly', 'new')

    to = '/home/ali/PycharmProjects/si/data'
    # to = os.path.join(to, 'p0' + str(p), 'daily')
    to = os.path.join(to, 'p0' + str(p), 'daily', 'new')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # MOTION
    modality = 'motion'
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    freq = 'D'

    df_01 = df.resample(freq).sum().astype(int)
    df_01.rename(columns={'motion': 'motion-sum'}, inplace=True)

    df_02 = df.resample(freq).apply(utils.get_nonzero_daytime_ratio)
    df_02.rename(columns={'motion': 'motion-ratio'}, inplace=True)

    df_03 = df.resample(freq).apply(utils.get_nonzero_mean)
    df_03.rename(columns={'motion': 'motion-mean'}, inplace=True)

    df_04 = df.resample(freq).max().astype(int)
    df_04.rename(columns={'motion': 'motion-max'}, inplace=True)

    df_05 = df.resample(freq).apply(utils.get_max_timestamp)
    df_05.rename(columns={'motion': 'motion-max-timestamp'}, inplace=True)

    _df_ = pd.concat([
        df_01,
        df_02,
        df_03,
        df_04,
        df_05,
    ], axis=1)

    _df_ = _df_.reindex(get_complete_date_range(_df_, freq))
    _df_.index.name = 'timestamp'
    _df_.fillna(0, inplace=True)
    _df_.reset_index(inplace=True)

    _df_.to_csv(os.path.join(to, modality + '.csv'), index=False)
    print(p, modality, '\t', df.shape, _df_.shape)


    # break