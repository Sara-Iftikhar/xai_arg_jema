"""
=====================
sul1 cross validation
=====================
"""

import warnings

def warn(*arg, **kwargs):
    pass

warnings.warn = warn

from ai4water import Model

from sul1_utils import sul1_data

x, y, input_features, output_features = sul1_data()

model = Model(model=   {
            "XGBRegressor": {
                "n_estimators": 5,
                "learning_rate": 0.33336666666666664,
                "booster": "gblinear",
                "random_state": 313
            }
        },


    x_transformation= [
            {
                "method": "zscore",
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
                "method": "robust",
                "features": [
                    "tide_cm"
                ]
            },
            {
                "method": "pareto",
                "features": [
                    "sal_psu"
                ]
            },
            {
                "method": "robust",
                "features": [
                    "pcp_mm"
                ]
            },
            {
                "method": "minmax",
                "features": [
                    "air_p_hpa"
                ]
            }
        ],
    y_transformation=  [
            {
                "method": "log",
                "features": [
                    "sul1_coppml"
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
