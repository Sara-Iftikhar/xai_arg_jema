"""
=======================
aac shap
=======================
"""

# %%
# model building
# ---------------

import numpy as np
import pandas as pd
from ai4water import Model
import matplotlib.pyplot as plt
from aac_utils import aac_data, get_fitted_model

x, y, input_features, output_features = aac_data()

np.set_printoptions(suppress=True, linewidth=200)

# %%

model = Model(
    model= {
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
    split_random = False,
    cross_validator= {"TimeSeriesSplit": {"n_splits": 10}},
    verbosity=0,
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

model.fit(x, y)

# %%
from ai4water.postprocessing.explain import ShapExplainer
import shap
print(shap.__version__)

#%%

class MyShapExpaliner(ShapExplainer):
    def summary_plot(
            self,
            plot_type: str = None,
            name: str = "summary_plot",
            **kwargs
    ):
        """
        Plots the `summary <https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html#shap.summary_plot>`_
        plot of SHAP package.
        Arguments:
            plot_type : str,
                either "bar", or "violen" or "dot"
            name:
                name of saved file
            kwargs:
                any keyword arguments to shap.summary_plot
        """

        def _summary_plot(_shap_val, _data, _features, _name):
            plt.close('all')

            shap.summary_plot(_shap_val, _data, show=False, plot_type=plot_type,
                              feature_names=_features,
                              **kwargs)
            # if self.save:
            #     plt.savefig(os.path.join(self.path, _name + " _bar"), dpi=300,
            #                 bbox_inches="tight")
            # if self.show:
            #     plt.show()

            return

        shap_vals = self.shap_values
        if isinstance(shap_vals, list) and len(shap_vals) == 1:
            shap_vals = shap_vals[0]

        data = self.data

        if self.single_source:
            if data.ndim == 3:
                assert shap_vals.ndim == 3

                for lookback in range(data.shape[1]):

                    _summary_plot(shap_vals[:, lookback],
                                  _data=data[:, lookback],
                                  _features=self.features,
                                  _name=f"{name}_{lookback}")
            else:
                _summary_plot(_shap_val=shap_vals, _data=data,
                              _features=self.features, _name=name)
        else:
            # data is a list of data sources
            for idx, _data in enumerate(data):
                if _data.ndim == 3:
                    for lb in range(_data.shape[1]):
                        _summary_plot(_shap_val=shap_vals[idx][:, lb],
                                      _data=_data[:, lb],
                                      _features=self.features[idx],
                                      _name=f"{name}_{idx}_{lb}")
                else:
                    _summary_plot(_shap_val=shap_vals[idx], _data=_data,
                                  _features=self.features[idx], _name=f"{name}_{idx}")

        return

# %%

explainer = MyShapExpaliner(model=model,
                         data=x,
                         train_data=x,
                         feature_names=model.input_features,
                          #save=False
                         )
print(explainer.explainer)

#%%

explainer.plot_shap_values()

#%%
#
# explainer.waterfall_plot_single_example(27)
#
# #%%
#
# explainer.waterfall_plot_single_example(28)
#
# #%%
#
# explainer.waterfall_plot_single_example(29)
#
# #%%
#
# explainer.waterfall_plot_single_example(30)
#
# #%%
#
# explainer.waterfall_plot_single_example(31)
#
# #%%
#
# explainer.waterfall_plot_single_example(32)
#
# #%%
#
# explainer.waterfall_plot_single_example(33)

#%%

#explainer.beeswarm_plot()

#%%

#explainer.heatmap()

#%%

explainer.summary_plot(plot_type="dot")

#%%

explainer.summary_plot(plot_type="bar")

#%%

explainer.summary_plot(plot_type="violin")

#%%

explainer.dependence_plot_all_features()

#%%

# explainer.scatter_plot_all_features()

#%% md

## Kernel Explainer
# ------------------

#%%

explainer = MyShapExpaliner(model=model,
                         data=x,
                         train_data=x,
                         feature_names=model.input_features,
                          explainer="KernelExplainer",
                         # save=False
                         )
print(explainer.explainer)

#%%

explainer.plot_shap_values()

#%%

#explainer.beeswarm_plot()

#%%

explainer.summary_plot(plot_type="bar")

#%%

explainer.summary_plot(plot_type="dot")

#%%

explainer.summary_plot(plot_type="violin")

#%%

#explainer.heatmap()

#%%

#explainer.waterfall_plot_single_example(27)
explainer.dependence_plot_all_features()

#%%

#explainer.scatter_plot_all_features()
