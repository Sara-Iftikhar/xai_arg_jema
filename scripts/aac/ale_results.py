"""
===============
aac ale
===============
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import imshow

from aac_utils import get_fitted_model

# %%
model = get_fitted_model()

# %%
train_df = pd.read_csv("../train_aac_rand.csv", index_col="Unnamed: 0")
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

print(train_x.shape, train_y.shape)

# %%

test_df = pd.read_csv("../test_aac_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]
print(test_x.shape, test_y.shape)

#%%

#%%

import PyALE
from PyALE import ale

#%% md

### sal_psu

#%%

## 1D - continuous - no CI
ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["sal_psu"], grid_size=50, include_CI=False
)

#%%

## 1D - continuous
_ = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["sal_psu"], grid_size=50, include_CI=True, C=0.95
)

#%%

_ = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["sal_psu"], grid_size=500, include_CI=False
)

#%% md

### pcp_mm

#%%

## 1D - continuous - no CI
_ = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["pcp_mm"], grid_size=50, include_CI=False
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["pcp_mm"], grid_size=50, include_CI=True, C=0.95
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["pcp_mm"], grid_size=500, include_CI=False
)

#%% md

### tide_cm

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["tide_cm"], grid_size=50, include_CI=False
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["tide_cm"], grid_size=50, include_CI=True, C=0.95
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["tide_cm"], grid_size=500, include_CI=False
)

#%% md

### wat_temp_c

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wat_temp_c"], grid_size=50, include_CI=False
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wat_temp_c"], grid_size=50, include_CI=True, C=0.95
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wat_temp_c"], grid_size=500, include_CI=False
)

#%% md

### air_p_hpa

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["air_p_hpa"], grid_size=50, include_CI=False
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["air_p_hpa"], grid_size=50, include_CI=True, C=0.95
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["air_p_hpa"], grid_size=500, include_CI=False
)

#%% md

### wind_speed_mps

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wind_speed_mps"], grid_size=50, include_CI=False
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wind_speed_mps"], grid_size=50, include_CI=True, C=0.95
)

#%%

ale_eff = ale(
    X=pd.DataFrame(train_x, columns=model.input_features),
    model=model, feature=["wind_speed_mps"], grid_size=500, include_CI=False
)

#%%

features=["air_p_hpa", "wat_temp_c"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["wind_speed_mps", "wat_temp_c"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["pcp_mm", "wat_temp_c"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["sal_psu", "wat_temp_c"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["tide_cm", "wat_temp_c"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["air_p_hpa", "tide_cm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["wind_speed_mps", "tide_cm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["pcp_mm", "tide_cm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["sal_psu", "tide_cm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["air_p_hpa", "sal_psu"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["wind_speed_mps", "sal_psu"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["pcp_mm", "sal_psu"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["air_p_hpa", "pcp_mm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["wind_speed_mps", "pcp_mm"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%%

features=["air_p_hpa", "wind_speed_mps"]
ale_eff = ale(X=pd.DataFrame(train_x, columns=model.input_features),
              model=model,
              feature=features,
              grid_size=500,
              plot=False)

X, Y = np.meshgrid(ale_eff.columns, ale_eff.index)
plt.close('all')
_ = imshow(ale_eff.values,
       origin="lower",
       extent=[X.min(), X.max(), Y.min(), Y.max()],
           colorbar=True,
           aspect="auto",
           xlabel=features[1],
           ylabel=features[0],
           cmap="Blues",
#            vmin=0,
#            vmax=1e7,
          )

#%% md
