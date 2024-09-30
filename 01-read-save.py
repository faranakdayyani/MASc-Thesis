import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

p = 8
root = '/home/ali/PycharmProjects/si-exploratory/si-data'
root = os.path.join(root, 'p0' + str(p))

to = '/home/ali/PycharmProjects/si/data'
to = os.path.join(to, 'p0' + str(p), 'raw')

dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
p_row = dates[dates['participant'] == 'p0' + str(p)]
start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
print(start_date, end_date)

# STEP
# modalities = ['wStep', 'Step', 'Pstep']

# ACCELERATION
# modalities = ['wAcceleration', 'Acceleration', 'Pacceleration']

# HEARTRATE
# modalities = ['wHeartrate', 'Heartrate']

# POSITION
# modalities = ['Position']

modalities = ['wStep']

for modality in modalities:
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df['datetime'] = pd.to_datetime(df['t'], format='%y-%m-%d-%H:%M:%S.%f')

    df = df.drop_duplicates(subset='datetime')
    df.sort_values(by='datetime', inplace=True, ascending=True)

    df['date'] = pd.to_datetime(df['datetime'], format='%d-%b-%y')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # STEP
    df['step'] = df['s'].astype(int)
    df = df[['datetime', 'step']]

    # ACCELERATION
    # df['x'] = df['x'].round(4)
    # df['y'] = df['y'].round(4)
    # df['z'] = df['z'].round(4)
    # df = df[['datetime', 'x', 'y', 'z']]

    # HEARTRATE
    # df['heartrate'] = df['h'].astype(int)
    # df = df[['datetime', 'heartrate']]

    # POSITION
    # df['latitude'] = df['a'].round(6)
    # df['longitude'] = df['o'].round(6)
    # df['speed'] = df['s'].astype(int)
    # df = df[['datetime', 'latitude', 'longitude', 'speed']]

    df.to_csv(os.path.join(to, modality + '.csv'), index=False)
    print(modality, '\t', df.shape)
