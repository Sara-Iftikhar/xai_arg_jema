
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai4water.datasets import busan_beach
from easy_mpl import ridge, hist

from ai4water.preprocessing import DataSet


def read_data(file_name, inputs= None, target='ecoli',
              power_transform_target=True):

    fpath = os.path.join(os.path.dirname(__file__), "data", file_name)
    df = pd.read_excel(fpath, index_col="Date_Time2")
    df.index = pd.to_datetime(df.index)

    default_inputs = ['wat_temp_c', 'tide_cm', 'sal_psu', 'pcp_mm', 'wind_speed_mps', 'air_p_hpa',
    'rel_hum']

    default_targets = [col for col in df.columns if col not in default_inputs]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    df = df[inputs + target]

    if power_transform_target:
        df[target] = np.power(10, df[target].values)

    return df


def make_whole_data(target,
                    reindex=True,
                    version='new',
                    remove_humidity=True,
                    power_transform_target=True,
                    ):
    data1 = busan_beach(inputs=['wat_temp_c', 'tide_cm', 'sal_psu', 'pcp_mm', 'wind_speed_mps',
                                'air_p_hpa', 'rel_hum'], target=target)

    assert version in ['new', 'old']

    if version == 'new':
        data2 = read_data(target=target, file_name="KarachiData_new.xlsx",
                          power_transform_target=power_transform_target)
        data3 = read_data(target=target, file_name="BalochistanData_new.xlsx",
                          power_transform_target=power_transform_target)
    else:
        data2 = read_data(target=target, file_name="KarachiData_old.xlsx",
                          power_transform_target=power_transform_target)
        data3 = read_data(target=target, file_name="BalochistanData_old.xlsx",
                          power_transform_target=power_transform_target)

    data= pd.concat([data1, data2, data3])

    data['pcp_mm'] = data['pcp_mm'].interpolate()

    if reindex:
        data = data.reset_index()
        # remove the "index" column which is inserted when we reset_index
        data.pop("index")

    if remove_humidity:
        data.pop("rel_hum")

    return data


def ridge_plot(arg, version):

    target = f'{arg}_coppml'
    data = make_whole_data(target=target, version=version)

    dataset = DataSet(data, train_fraction=0.8,
                      val_fraction=0.0,
                      split_random=False)

    train_x, train_y = dataset.training_data()
    test_x, test_y = dataset.test_data()

    ## random data

    train_rand = pd.read_csv(f'train_{arg}_rand.csv')
    test_rand = pd.read_csv(f'test_{arg}_rand.csv')

    train_y_rand = train_rand.iloc[:, -1].to_numpy()
    test_y_rand = test_rand.iloc[:, -1].to_numpy()

    total = [train_y_rand, test_y_rand, train_y.reshape(-1, ), test_y.reshape(-1, )]

    random = pd.DataFrame(total[0:2]).T
    seq = pd.DataFrame(total[2:]).T
    random.columns = ["Training", "Test"]
    seq.columns = ["Training", "Test"]

    ridge(random,
          # xlabel='train_random and test_random',
          # title=target,
          cmap='Blues',
          )

    ridge(seq,
          # xlabel='train_sequential and test_sequential',
          # title=target,
          cmap='Blues',
          )
    return