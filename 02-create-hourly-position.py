import os
import pandas as pd
import warnings
import numpy as np
import utils

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:

    geofences = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si/', 'notes-geofences' + '.csv'))
    home_lat, home_lon = (geofences[geofences['participant'] == 'p0' + str(p)]['latitude'].item(),
                          geofences[geofences['participant'] == 'p0' + str(p)]['longitude'].item())

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
    modalities = ['Position']

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
    df_01 = df[['speed']].resample(freq).count().round(2)
    df_01.rename(columns={'speed': 'count'}, inplace=True)

    df_02 = df[['speed']].resample(freq).mean().round(2)
    df_02.rename(columns={'speed': 'mean'}, inplace=True)

    df_03 = df[['speed']].resample(freq).max().round(2)
    df_03.rename(columns={'speed': 'max'}, inplace=True)

    df_04 = df[['speed']].resample(freq).apply(utils.get_time_difference).round(2)
    df_04.rename(columns={'speed': 'duration'}, inplace=True)

    df_05 = df[['latitude', 'longitude']].resample(freq).apply(
        lambda x: utils.get_maximum_distance(x, home_lat, home_lon)).round(2)
    df_05 = df_05.to_frame()
    df_05.rename(columns={df_05.columns[0]: 'distance-maximum'}, inplace=True)

    df_06 = df[['latitude', 'longitude']].resample(freq).apply(
        lambda x: utils.get_travelled_distance(x, home_lat, home_lon)).round(2)
    df_06 = df_06.to_frame()
    df_06.rename(columns={df_06.columns[0]: 'distance-travelled'}, inplace=True)

    _df_ = pd.concat([
        df_01,
        df_02,
        df_03,
        df_04,
        df_05,
        df_06], axis=1)

    _df_ = _df_.reindex(utils.get_complete_date_range(df_01, freq))
    _df_.index.name = 'timestamp'
    _df_.fillna(0, inplace=True)
    # _df_['motion'] = _df_['motion'].astype(int)
    _df_.reset_index(inplace=True)

    _df_.to_csv(os.path.join(to, 'position' + '.csv'), index=False)
    print(p, modality, '\t', df.shape, _df_.shape)

