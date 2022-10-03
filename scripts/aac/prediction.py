"""
==========================
aac prediction
==========================
"""
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import numpy as np
import pandas as pd
from ai4water import Model
from ai4water.preprocessing import DataSet
from easy_mpl import plot, bar_chart
import matplotlib.pyplot as plt
from SeqMetrics import RegressionMetrics

from aac_utils import make_whole_data

np.set_printoptions(suppress=True, linewidth=200)

# %%

model = Model(
    model= {
            "CatBoostRegressor": {
                "iterations": 500,
                "learning_rate": 0.49999999999999994,
                "l2_leaf_reg": 0.5,
                "model_size_reg": 3.1912231399066187,
                "rsm": 0.8001459176683743,
                "border_count": 1032,
                "feature_border_type": "UniformAndQuantiles",
                "logging_level": "Silent",
                "random_seed": 891
            }
        },
    x_transformation=[
            {
                "method": "quantile",
                "features": [
                    "wat_temp_c"
                ]
            },
            {
                "method": "robust",
                "features": [
                    "tide_cm"
                ]
            },
            {
                "method": "log",
                "features": [
                    "sal_psu"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "sqrt",
                "features": [
                    "pcp_mm"
                ],
                "treat_negatives": True
            },
            {
                "method": "scale",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "quantile",
                "features": [
                    "air_p_hpa"
                ]
            }
        ],
    y_transformation=[
            {
                "method": "log",
                "features": [
                    "aac_coppml"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            }
        ],
    seed=891,
    input_features=[
            "wat_temp_c",
            "tide_cm",
            "sal_psu",
            "pcp_mm",
            "wind_speed_mps",
            "air_p_hpa"
        ],
    output_features=[
            "aac_coppml"
        ],
)

# %%
train_df = pd.read_csv("../train_aac_rand.csv", index_col="Unnamed: 0")
print(train_df.shape)

# %%
train_df.head()

# %%
train_x, train_y = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

# %%
_ = model.fit(x=train_x.values, y=train_y.values)


# %%

#%% md
# Training data
# ----------------

#%%

train_true, train_pred = model.predict(x=train_x.values, y=train_y.values, return_true=True)
print(np.mean(train_true), np.mean(train_pred))

#%%

metrics = RegressionMetrics(train_true, train_pred).calculate_all()

for metric in ["r2", "nse", "nrmse", "rmsle", "mape", "pbias", "rmse"]:
    print(metric, metrics[metric])

#%%

_, ax = plt.subplots(figsize=(10, 5))
plot(train_true, '--.', show=False, label="True")
plot(train_pred, '--.', label="Predicted")

#%%

_, _ = plt.subplots(figsize=(10, 5))
plot(train_true, '--.', show=False, label="True", logy=True)
plot(train_pred, '--.', label="Predicted")

#%%

np.argmax(train_true), train_true[np.argmax(train_true)]

#%%

print(train_pred[275])

#%%

print(train_x.iloc[275])

#%%

p = model.predict(x=train_x.iloc[275].values.reshape(1, -1))

#%%

print(p)

#%%

_, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(10, 8), sharex="all")
plot(train_x["pcp_mm"].values[0:33], '--.', show=False, ax=ax1, label="pcp_mm")
plot(train_x["sal_psu"].values[0:33], '--.', show=False, ax=ax2, label="sal_psu")
plot(train_x["tide_cm"].values[0:33], '--.', show=False, ax=ax3, label="tide_cm")
plot(train_x["wat_temp_c"].values[0:33], '--.', show=False, ax=ax4, label="wat_temp_c")
plot(train_true[0:33], '--.', show=False, label="True", logy=True, ax=ax5)
plot(train_pred[0:33], '--.', label="Predicted", ax=ax5, show=False)

#%% md
# Test Data
# -----------

# %%
test_df = pd.read_csv("../test_aac_rand.csv", index_col="Unnamed: 0")
test_x, test_y = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]

#%%

print(test_df.shape)

# %%

test_df.head()

#%%

test_true, test_pred = model.predict(x=test_x.values, y=test_y.values, return_true=True)
print(np.mean(test_true), np.mean(test_pred))

#%%

metrics = RegressionMetrics(test_true, test_pred).calculate_all()

for metric in ["r2", "nse", "nrmse", "rmsle", "mape", "pbias"]:
    print(metric, metrics[metric])

#%%

_, _ = plt.subplots(figsize=(10, 5))
plot(test_true, '--.', show=False, label="True")
_ = plot(test_pred, '--.', label="Predicted")

#%%

_, _ = plt.subplots(figsize=(10, 5))
plot(test_true, '--.', show=False, label="True", logy=True)
_ = plot(test_pred, '--.', label="Predicted")

#%%

_, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(10, 12), sharex="all")
plot(test_x["pcp_mm"].values[0:33], '--.', show=False, ax=ax1, label="pcp_mm")
plot(test_x["sal_psu"].values[0:33], '--.', show=False, ax=ax2, label="sal_psu")
plot(test_x["tide_cm"].values[0:33], '--.', show=False, ax=ax3, label="tide_cm")
plot(test_x["wat_temp_c"].values[0:33], '--.', show=False, ax=ax4, label="wat_temp_c")
plot(test_x["wind_speed_mps"].values[0:33], '--.', show=False, ax=ax5, label="wind_speed_mps")
plot(test_x["air_p_hpa"].values[0:33], '--.', show=False, ax=ax6, label="air_p_hpa")
plot(test_true[0:33], '--.', show=False, label="True", logy=True, ax=ax7)
plot(test_pred[0:33], '--.', label="Predicted", ax=ax7, show=False)

#%%

np.mean(train_pred), np.mean(test_pred)

#%%

fi = model._model.feature_importances_
bar_chart(fi, labels=model.input_features, sort=True)

# %% md
## all data
# ---------

#%%


all_data = make_whole_data(target="aac_coppml")
print(all_data.shape)

#%%

all_data.head()

#%%

ds = DataSet(data=all_data, train_fraction=1.0, val_fraction=0.0)
all_x, all_y = ds.training_data()

#%%

all_t, all_p = model.predict(x=all_x, y=all_y, return_true=True, process_results=False)

#%%

_, _ = plt.subplots(figsize=(18, 8))
_ = plot(all_p, '--.', label="Predicted", color="silver", logy=True, show=False)
ax = plot(all_t, '.', label="True", logy=True,
     ax=ax, show=False)
ax.legend(fontsize=20, markerscale=4)
ax.set_xlabel("Number of Exmaples", fontsize=20)
ax.set_ylabel("aac (Coppml)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)

#%%

metrics = RegressionMetrics(all_t, all_p).calculate_all()

for metric in ["r2", "nse", "nrmse", "rmsle", "mape", "pbias"]:
    print(metric, metrics[metric])

#%%

all_t, all_p = model.predict(x=all_data.values[:, 0:-1],
                             y=all_data.values[:, -1],
                             return_true=True, process_results=False)

#%%

_, ax = plt.subplots(figsize=(18, 6))
_ = plot(all_p, '--.', label="Predicted", color="silver", show=False)
plot(all_t, '.', label="True", xlabel="Number of Exmaples", ylabel="aac (Coppml)", ax=ax)

#%%

_, ax = plt.subplots(figsize=(18, 10))
_ = plot(all_p, '--.', label="Predicted", color="silver", logy=True, show=False)
plot(all_t, '.', label="True", logy=True, xlabel="Number of Exmaples", ylabel="aac (Coppml)", ax=ax)


#%%

_, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(14, 12), sharex="all")
plot(all_data["pcp_mm"].values, '--.', show=False, ax=ax1, label="pcp_mm")
plot(all_data["sal_psu"].values, '--.', show=False, ax=ax2, label="sal_psu")
plot(all_data["tide_cm"].values, '--.', show=False, ax=ax3, label="tide_cm")
plot(all_data["wat_temp_c"].values, '--.', show=False, ax=ax4, label="wat_temp_c")
plot(all_data["wind_speed_mps"].values, '--.', show=False, ax=ax5, label="wind_speed_mps")
plot(all_data["air_p_hpa"].values, '--.', show=False, ax=ax6, label="air_p_hpa")
plot(all_t, '.', show=False, label="True", logy=True, ax=ax7)
plot(all_p, '--', label="Predicted", ax=ax7, show=False)
