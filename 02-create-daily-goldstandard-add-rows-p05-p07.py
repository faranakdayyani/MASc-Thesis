import os
import pandas as pd
import warnings
import numpy as np
from io import StringIO

import utils
from utils import get_complete_date_range

warnings.filterwarnings('ignore')

# for p in range(1, 7 + 1):
for p in [5]:
    root = '/home/ali/PycharmProjects/si/data'
    root = os.path.join(root, 'p0' + str(p), 'raw')

    to = '/home/ali/PycharmProjects/si/data'
    to = os.path.join(to, 'p0' + str(p), 'daily')

    # GOLDSTANDARD
    modality = 'p0' + str(p) + '-sis'

    goldstandard = pd.read_csv(os.path.join(root, modality + '.csv'))

    goldstandard_date = pd.read_csv(os.path.join(to, 'goldstandard-date' + '.csv'))

    df = goldstandard.merge(goldstandard_date, right_on='goldstandard-date', left_on='timestamp', how='left')

    df = df.drop('timestamp_x', axis=1)
    df = df.rename(columns={'timestamp_y': 'timestamp',
                            'goldstandard-date': 'goldstandard-timestamp'})

    cols = df.columns[-2:].tolist() + df.columns[:-2].tolist()
    df = df[cols]

    df.to_csv(os.path.join(to, 'goldstandard' + '.csv'), index=False)
