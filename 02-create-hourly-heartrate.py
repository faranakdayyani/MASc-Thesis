import os
import pandas as pd
import warnings
import numpy as np
import utils

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:
    # geofences = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si/', 'notes-geofences' + '.csv'))
    # home_lat, home_lon = (geofences[geofences['participant'] == 'p0' + str(p)]['latitude'].item(),
    #                       geofences[geofences['participant'] == 'p0' + str(p)]['longitude'].item())

    root = '/home/ali/PycharmProjects/si/data'
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    to = os.path.join(to, 'p0' + str(p), 'hourly')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # POSITION
    modalities = ['wHeartrate']

    # df1 = pd.read_csv(os.path.join(root, modalities[0] + '.csv'))
    # df2 = pd.read_csv(os.path.join(root, modalities[1] + '.csv'))
    # print(df1.shape[0])
    # print(df2.shape[0])
    # print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())))
    # print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())) == df2.shape[0])

    modality = modalities[0]
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    freq = 'H'
    df_01 = df[['heartrate']].resample(freq).min().round(2)
    df_01.rename(columns={'heartrate': 'min'}, inplace=True)

    df_02 = df[['heartrate']].resample(freq).max().round(2)
    df_02.rename(columns={'heartrate': 'max'}, inplace=True)

    df_03 = df[['heartrate']].resample(freq).mean().round(2)
    # df_03.rename(columns={'heartrate': 'mean'}, inplace=True)

    df_04 = df[['heartrate']].resample(freq).std().round(2)
    df_04.rename(columns={'heartrate': 'std'}, inplace=True)

    _df_ = pd.concat([
        # df_01,
        # df_02,
        df_03,
        # df_04
    ], axis=1)

    _df_ = _df_.reindex(utils.get_complete_date_range(df_01, freq))
    _df_.index.name = 'timestamp'
    _df_.fillna(0, inplace=True)
    # _df_['motion'] = _df_['motion'].astype(int)
    _df_.reset_index(inplace=True)

    _df_.to_csv(os.path.join(to, 'heartrate' + '.csv'), index=False)
    print(p, modality, '\t', df.shape, _df_.shape)
