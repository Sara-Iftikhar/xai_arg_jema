"""
===============
aac permutation
===============
"""

import pandas as pd


from aac_utils import get_fitted_model, aac_data

# %%
model = get_fitted_model()

# %%

x, y, input_features, output_features = aac_data()

#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="r2",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart", sort=True)

#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="nse",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart", sort=True)


#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="rmsle",
                                    plot_type="boxplot")