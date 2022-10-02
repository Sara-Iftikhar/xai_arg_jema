"""
=========================
tetx sensitivity
=========================
"""


import pandas as pd

from tetx_utils import get_fitted_model, sobol_plots, sensitivity_plots, Model

# %%
model = get_fitted_model(Model)

# %%
train_df = pd.read_csv("../train_tetx_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%
test_df = pd.read_csv("../test_tetx_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)

import SALib
from ai4water.postprocessing._sa import morris_plots
print(SALib.__version__)

#%%


res = model.sensitivity_analysis(data=train_x.values,
                                 sampler="morris",
                                 analyzer=["sobol", "pawn", "morris", "rbd_fast"],
                                 sampler_kwds={"N": 20000}
                                )

#%%

morris_plots(res["morris"], show=True)

#%%

sobol_plots(res["sobol"])

#%%

sensitivity_plots("rbd_fast", res["rbd_fast"], show=True)

#%%

sensitivity_plots("pawn", res["pawn"], show=True)

#%%

res = model.sensitivity_analysis(data=train_x.values,
                                 sampler="fast_sampler",
                                 analyzer=["fast"],
                                 sampler_kwds={"N": 20000}
                                )

#%%