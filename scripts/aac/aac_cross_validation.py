"""
=====================
aac cross validation
=====================
"""

import warnings

def warn(*arg, **kwargs):
    pass

warnings.warn = warn

from ai4water import Model

from aac_utils import aac_data

x, y, input_features, output_features = aac_data()

model = Model(model=   {
            "XGBRegressor": {
                "n_estimators": 5,
                "learning_rate": 0.0001,
                "booster": "gblinear",
                "random_state": 313
            }
        },


    x_transformation= [
            {
                "method": "pareto",
                "features": [
                    "wat_temp_c"
                ]
            },
            {
                "method": "quantile_normal",
                "features": [
                    "sal_psu"
                ],
                "n_quantiles": 40
            },
            {
                "method": "quantile",
                "features": [
                    "pcp_mm"
                ],
                "n_quantiles": 40
            },
            {
                "method": "sqrt",
                "features": [
                    "wind_speed_mps"
                ],
                "treat_negatives": True
            },
            {
                "method": "pareto",
                "features": [
                    "air_p_hpa"
                ]
            }
        ],
    y_transformation=   [
            {
                "method": "zscore",
                "features": [
                    "aac_coppml"
                ]
            }
        ],


                seed=313,
                output_features=output_features,
                input_features=input_features,
                split_random = False,
                cross_validator= {"TimeSeriesSplit": {"n_splits": 10}},
                verbosity=0,
                )


print(model.cross_val_score(x=x, y=y, scoring='r2'))
print(model.cross_val_score(x=x, y=y, scoring='r2_score'))
print(model.cross_val_score(x=x, y=y, scoring='rmse'))
print(model.cross_val_score(x=x, y=y, scoring='rmsle'))
print(model.cross_val_score(x=x, y=y, scoring='pbias'))
