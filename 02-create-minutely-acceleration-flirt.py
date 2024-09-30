import os
import pandas as pd
import warnings
import numpy as np
import utils
import flirt

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:
    # geofences = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si/', 'notes-geofences' + '.csv'))
    # home_lat, home_lon = (geofences[geofences['participant'] == 'p0' + str(p)]['latitude'].item(),
    #                       geofences[geofences['participant'] == 'p0' + str(p)]['longitude'].item())

    root = '/home/ali/PycharmProjects/si/data'
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    to = os.path.join(to, 'p0' + str(p), 'minutely')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # ACCELERATION
    modalities = ['wAcceleration']

    # df1 = pd.read_csv(os.path.join(root, modalities[0] + '.csv'))
    # df2 = pd.read_csv(os.path.join(root, modalities[1] + '.csv'))
    # print(df1.shape[0])
    # print(df2.shape[0])
    # print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())))
    # print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())) == df2.shape[0])

    modality = modalities[0]
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df.head(5)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    freq = '1T'

    # df_xyz = pd.concat([
    #     df['x'].resample(freq).mean(),
    #     df['y'].resample(freq).mean(),
    #     df['z'].resample(freq).mean()
    # ], axis=1)

    df_xyz = pd.concat([
        df['x'].resample(freq).first(),
        df['y'].resample(freq).first(),
        df['z'].resample(freq).first()
    ], axis=1)

    full_range = pd.date_range(start_date, end_date + pd.Timedelta(hours=23, minutes=59), freq='1T')
    df_xyz = df_xyz.reindex(full_range, fill_value=0)

    df_xyz = df_xyz.reset_index()
    df_xyz.rename(columns={'index': 'datetime'}, inplace=True)

    df_xyz['date'] = df_xyz['datetime'].dt.date
    df_xyz = df_xyz[['datetime', 'date', 'x', 'y', 'z']]

    df_xyz.fillna(0, inplace=True)

    df_xyz.to_csv(os.path.join(to, modality + '.csv'), index=False)



    dates = df_xyz['date']
    date_uniq = np.unique(dates).tolist()


    i = 2

    rows = []
    for date in date_uniq:

        _ = df_xyz[df_xyz['date'] == date]


        if _.shape[0] > 0:
            _dropped = _.drop(['date', 'datetime'], axis=1)

            row = flirt.get_acc_features(_dropped, data_frequency=1440)

            row['date'] = date

            print('Date: ', date, _.shape, row.shape)

            rows.append(row)

        # i += 1
        # if i > 3:
        #     break


    flirt_df = pd.concat(rows, axis=0)
    # flirt_df.to_csv(os.path.join(to, 'flirt' + '.csv'), index=False)

