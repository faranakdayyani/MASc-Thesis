import os
import csv
import pandas as pd
import json
import numpy as np



p = 7
root = '/home/ali/PycharmProjects/si-exploratory'
participant = 'participantuoft0' + str(p) + '@gmail.com'
participant0x = 'participant0' + str(p)

dates = pd.read_csv(os.path.join(root, 'notes-p01-07.csv'))
p_row = dates[dates['participant'] == 'p0' + str(p)]
start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')


files = os.listdir(os.path.join(root, participant))

modalities = [
    'wAcceleration',
    'Acceleration',
    'Pacceleration',

    'wHeartrate',
    'Heartrate',

    'wStep',
    'Step',
    'Pstep',

    'Position',
]
# modality = modalities[2]

for modality in modalities:

    print(modality)

    _files_ = [file for file in files if file.startswith(modality)]
    _list_ = []

    for file in _files_:
        # for file in [this_file]:

        csv = pd.read_csv(os.path.join(root, participant, file))
        json_str = csv['jsonData'].item()
        json_data = json.loads(json_str)
        # print(file, '---', csv.time)
        # print(json_data)
        _list_ += json_data
        # break

    sorted_list = sorted(_list_, key=lambda x: x['t'])

    # Split 't' into separate components and prepare a new list with the structured data
    structured_data = []
    if modality.__contains__('cceleration'):
        for item in sorted_list:
            year, month, day, time = item['t'].split('-')
            hour, minute, second = time.split(':')
            structured_data.append({
                'year': '20' + year,  # Assuming all dates are in the 2000s
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': round(float(second), 3),

                'x': round(item['x'], 4),
                'y': round(item['y'], 4),
                'z': round(item['z'], 4)
            })
    elif modality.__contains__('eartrate'):
        for item in sorted_list:
            year, month, day, time = item['t'].split('-')
            hour, minute, second = time.split(':')
            structured_data.append({
                'year': '20' + year,  # Assuming all dates are in the 2000s
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': round(float(second), 3),

                'heartrate': round(item['h'])
            })
    elif modality.__contains__('tep'):
        for item in sorted_list:
            year, month, day, time = item['t'].split('-')
            hour, minute, second = time.split(':')
            structured_data.append({
                'year': '20' + year,  # Assuming all dates are in the 2000s
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': round(float(second), 3),

                'step': round(item['s'])
            })
    elif modality.__contains__('osition'):
        for item in sorted_list:
            year, month, day, time = item['t'].split('-')
            hour, minute, second = time.split(':')
            structured_data.append({
                'year': '20' + year,  # Assuming all dates are in the 2000s
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': round(float(second), 3),

                'latitude': item['a'],
                'longitude': item['o'],
                'speed': round(item['s'], 4),
            })

    # Convert to DataFrame
    df = pd.DataFrame(structured_data)

    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df = df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]

    # Set datetime as the index
    # df.set_index('datetime', inplace=True)

    # Create a 'date' column by combining year, month, day
    # df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.date


    # df['_week_'] = df['datetime'].dt.isocalendar().week
    # df['_day_'] = df['datetime'].dt.date
    # df['_period_'] = df['datetime'].apply(categorize_time_of_day)
    # df['_hour_'] = df['datetime'].dt.hour

    # Now you can resample based on your window length. Here's an example for a 1-day window.
    # Use 'T' for minutes, 'H' for hours, 'D' for days, 'W' for weeks, etc.
    # resampled_df = df.resample('w').sum()

    df = df.drop_duplicates(subset='datetime')

    df['date'] = pd.to_datetime(df['datetime'], format='%d-%b-%y')
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    df.to_csv(os.path.join(root, 'participants', participant0x, 'temp', modality + '.csv'), index=False)

