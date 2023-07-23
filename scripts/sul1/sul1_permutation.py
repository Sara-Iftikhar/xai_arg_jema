"""
================
sul1 permutation
================
"""

from ai4water import Model

from sul1_utils import get_fitted_model, sul1_data

# %%
model = get_fitted_model(Model)

# %%

x, y, input_features, output_features = sul1_data()

#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="r2",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart")

#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="nse",
                                    plot_type="boxplot")

#%%

pimp.plot_1d_pimp("bar_chart")


#%%

pimp = model.permutation_importance(x=x,
                                    y=y,
                                    n_repeats=100,
                                    scoring="rmsle",
                                    plot_type="boxplot")