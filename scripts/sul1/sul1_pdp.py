"""
==============
sul1 pdp
==============
"""

import pandas as pd

from utils import get_fitted_model, Model, PartialDependencePlot1

# %%
model = get_fitted_model(Model)

# %%
train_df = pd.read_csv("../train_sul1_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%
test_df = pd.read_csv("../test_sul1_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)


pdp = PartialDependencePlot1(
    model=model.predict,
    data=train_x,
    feature_names=model.input_features,
    num_points=100,
    save=False,
    path=model.path
)

#%% md

### sal_psu

#%%

_ = pdp.plot_1d("sal_psu")

#%%

pdp.plot_1d("sal_psu", ice=False)

#%%

pdp.plot_1d("sal_psu", ice_only=True, ice_color="green")

#%% md

### pcp_mm

#%%

_ = pdp.plot_1d("pcp_mm")

#%%

_ = pdp.plot_1d("pcp_mm", ice=False)

#%%

_ = pdp.plot_1d("pcp_mm", show_ci=False, ice_only=True, ice_color="green")

#%% md

### tide_cm

#%%

_ = pdp.plot_1d("tide_cm")

#%%

_ = pdp.plot_1d("tide_cm", ice=False)

#%%

_ = pdp.plot_1d("tide_cm", show_ci=False, ice_only=True, ice_color="green")

#%% md

### wat_temp_c

#%%

_ = pdp.plot_1d("wat_temp_c")

#%%

_ = pdp.plot_1d("wat_temp_c", ice=False)

#%%

_ = pdp.plot_1d("wat_temp_c", show_ci=False, ice_only=True, ice_color="green")

#%% md

### air_p_hpa

#%%

_ = pdp.plot_1d("air_p_hpa")

#%%

_ = pdp.plot_1d("air_p_hpa", ice=False)

#%%

_ = pdp.plot_1d("air_p_hpa", show_ci=False, ice_only=True, ice_color="green")

#%% md

### wind_speed_mps

#%%

_ = pdp.plot_1d("wind_speed_mps")

#%%

_ = pdp.plot_1d("wind_speed_mps", ice=False)

#%%

_ = pdp.plot_1d("wind_speed_mps", show_ci=False, ice_only=True, ice_color="green")

#%% md

## interaction

#%%

# _ = pdp.plot_interaction(["tide_cm", "pcp_mm"], cmap="Blues")
#
# #%%
#
# pdp.plot_interaction(["tide_cm", "wat_temp_c"], cmap="Blues")
#
# #%%
#
# pdp.plot_interaction(["tide_cm", "sal_psu"], cmap="Blues")
#
# #%%
#
# pdp.plot_interaction(["pcp_mm", "wat_temp_c"], cmap="Blues")
#
# #%%
#
# pdp.plot_interaction(["pcp_mm", "sal_psu"], cmap="Blues")
#
# #%%
#
# pdp.plot_interaction(["wat_temp_c", "sal_psu"], cmap="Blues")
#
# #%%
#
# _ = pdp.nd_interactions()