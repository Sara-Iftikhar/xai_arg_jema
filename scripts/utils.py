
import os
import numpy as np
import pandas as pd
from ai4water.datasets import busan_beach


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
