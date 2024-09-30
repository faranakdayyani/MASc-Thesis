import os
import pandas as pd
import warnings
import numpy as np

from utils import get_complete_date_range

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [8]:


    root = '/home/ali/PycharmProjects/si/data'
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    to = os.path.join(to, 'p0' + str(p), 'hourly')

    dates = pd.read_csv(os.path.join('/home/ali/PycharmProjects/si', 'notes-dates.csv'))
    p_row = dates[dates['participant'] == 'p0' + str(p)]
    start_date = pd.to_datetime(p_row.start.item(), format='%d-%b-%y')
    end_date = pd.to_datetime(p_row.end.item(), format='%d-%b-%y')
    print(start_date, end_date)

    # STEP
    modalities = ['wStep', 'Step', 'Pstep']

    df1 = pd.read_csv(os.path.join(root, modalities[0] + '.csv'))
    df2 = pd.read_csv(os.path.join(root, modalities[1] + '.csv'))
    print(df1.shape[0])
    print(df2.shape[0])
    print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())))
    print(len(np.intersect1d(df1['datetime'].unique(), df2['datetime'].unique())) == df2.shape[0])

    modality = modalities[0]
    df = pd.read_csv(os.path.join(root, modality + '.csv'))

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    freq = 'H'
    _df_ = df.resample(freq).sum()

    _df_ = _df_.reindex(get_complete_date_range(_df_, freq))
    _df_.index.name = 'timestamp'
    _df_.fillna(0, inplace=True)
    _df_['step'] = _df_['step'].astype(int)
    _df_.reset_index(inplace=True)

    _df_.to_csv(os.path.join(to, 'step' + '.csv'), index=False)
    print(p, modality, '\t', df.shape, _df_.shape)


