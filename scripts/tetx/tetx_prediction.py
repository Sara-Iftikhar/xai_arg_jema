"""
================
tetx prediction
=================
"""


import ai4water
ai4water.__version__

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import plot, bar_chart
from ai4water.preprocessing import DataSet
np.__version__

#%%


from SeqMetrics import RegressionMetrics
from utils import get_fitted_model, Model

#%%

model = get_fitted_model(Model)

train_df = pd.read_csv("../train_tetx_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]
test_df = pd.read_csv("../test_tetx_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]