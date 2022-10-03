"""
===============
aac permutation
===============
"""

import pandas as pd


from aac_utils import get_fitted_model

# %%
model = get_fitted_model()


# %%
test_df = pd.read_csv("../test_aac_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)

#%%

pimp = model.permutation_importance(x=test_x,
                                    y=test_y,
                                    n_repeats=100,
                                    scoring="r2",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart", sort=True)

#%%

# pimp = model.permutation_importance(x=test_x,
#                                     y=test_y,
#                                     n_repeats=1000,
#                                     scoring="r2",
#                                     plot_type="boxplot")
#
# #%%
#
# pimp.plot_1d_pimp("bar_chart", sort=True)

#%%

pimp = model.permutation_importance(x=test_x,
                                    y=test_y,
                                    n_repeats=100,
                                    scoring="nse",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart", sort=True)

#%%

# pimp = model.permutation_importance(x=test_x,
#                                     y=test_y,
#                                     n_repeats=1000,
#                                     scoring="nse",
#                                     plot_type="boxplot")
#
# #%%
#
# pimp.plot_1d_pimp("bar_chart", sort=True)

#%%

pimp = model.permutation_importance(x=test_x,
                                    y=test_y,
                                    n_repeats=100,
                                    scoring="mse",
                                    plot_type="boxplot")

#%%
#
# pimp = model.permutation_importance(x=test_x,
#                                     y=test_y,
#                                     n_repeats=1000,
#                                     scoring="mse",
#                                     plot_type="boxplot")
#
# #%%
#
# pimp.plot_1d_pimp("bar_chart", sort=True)