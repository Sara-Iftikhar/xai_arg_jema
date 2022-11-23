"""
==========================
aac prediction
==========================
"""

from ai4water import Model
from ai4water.preprocessing import DataSet
from SeqMetrics import RegressionMetrics
from aac_utils import make_whole_data

data = make_whole_data("aac_coppml")

input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

dataset = DataSet(data,
                    train_fraction=1.0,
                  split_random=True,
                  seed=891)
train_x, train_y= dataset.training_data()
test_x, test_y= dataset.validation_data()

# %%

model = Model(
    model=  {
            "XGBRegressor": {
                "n_estimators": 31,
                "learning_rate": 0.4016665322343523,
                "booster": "gbtree",
                "random_state": 313
            }
        },
    x_transformation= [
            {
                "method": "sqrt",
                "features": [
                    "wat_temp_c"
                ],
                "treat_negatives": True
            },
            {
                "method": "quantile",
                "features": [
                    "tide_cm"
                ],
                "n_quantiles": 239
            },
            {
                "method": "scale",
                "features": [
                    "sal_psu"
                ]
            },
            {
                "method": "log10",
                "features": [
                    "pcp_mm"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "minmax",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "log2",
                "features": [
                    "air_p_hpa"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            }
        ],
    y_transformation= [
            {
                "method": "quantile",
                "features": [
                    "aac_coppml"
                ],
                "n_quantiles": 239
            }
        ],
    seed=891,
    input_features= input_features,
    output_features=output_features,
)

# %%
_ = model.fit(x=train_x, y=train_y)

#%% md
# Training data
# ----------------

#%%

train_true, train_pred = model.predict(x=train_x, y=train_y, return_true=True)

#%%

metrics = RegressionMetrics(train_true, train_pred).calculate_all()

for metric in ["r2", "r2_score", "nrmse", "rmsle", "mape", "pbias", "rmse"]:
    print(metric, metrics[metric])

#%%

#%% md
# Test data
# ----------------

#%%

test_true, test_pred = model.predict(x=test_x, y=test_y, return_true=True)

#%%

metrics = RegressionMetrics(test_true, test_pred).calculate_all()

for metric in ["r2", "r2_score", "nrmse", "rmsle", "mape", "pbias"]:
    print(metric, metrics[metric])

