"""
=====================
tetx cross validation
=====================
"""

import warnings

def warn(*arg, **kwargs):
    pass

warnings.warn = warn
from ai4water import Model

from tetx_utils import tetx_data

x, y, input_features, output_features = tetx_data()

model = Model(model=   {
            "XGBRegressor": {
                "n_estimators": 91,
                "learning_rate": 0.3889111111111111,
                "booster": "dart",
                "random_state": 313
            }
        },


    x_transformation= [
            {
                "method": "center",
                "features": [
                    "wind_speed_mps"
                ]
            },
            {
                "method": "log2",
                "features": [
                    "wat_temp_c"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "pareto",
                "features": [
                    "tide_cm"
                ]
            },
            {
                "method": "log10",
                "features": [
                    "sal_psu"
                ],
                "treat_negatives": True,
                "replace_zeros": True
            },
            {
                "method": "center",
                "features": [
                    "pcp_mm"
                ]
            },
            {
                "method": "quantile",
                "features": [
                    "air_p_hpa"
                ],
                "n_quantiles": 40
            }
        ],
    y_transformation=  [
            {
                "method": "log10",
                "features": [
                    "tetx_coppml"
                ],
                "treat_negatives": True,
                "replace_zeros": True
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
